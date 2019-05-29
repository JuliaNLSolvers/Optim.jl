abstract type Simplexer end

struct AffineSimplexer <: Simplexer
    a::Float64
    b::Float64
end
AffineSimplexer(;a = 0.025, b = 0.5) = AffineSimplexer(a, b)

function simplexer(S::AffineSimplexer, initial_x::Tx) where Tx
    n = length(initial_x)
    initial_simplex = Tx[copy(initial_x) for i = 1:n+1]
    for j = 1:n
        initial_simplex[j+1][j] = (1+S.b) * initial_simplex[j+1][j] + S.a
    end
    initial_simplex
end

abstract type NMParameters end

struct AdaptiveParameters <: NMParameters
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
end

AdaptiveParameters(;  α = 1.0, β = 1.0, γ = 0.75 , δ = 1.0) = AdaptiveParameters(α, β, γ, δ)
parameters(P::AdaptiveParameters, n::Integer) = (P.α, P.β + 2/n, P.γ - 1/2n, P.δ - 1/n)

struct FixedParameters <: NMParameters
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
end

FixedParameters(; α = 1.0, β = 2.0, γ = 0.5, δ = 0.5) = FixedParameters(α, β, γ, δ)
parameters(P::FixedParameters, n::Integer) = (P.α, P.β, P.γ, P.δ)

struct NelderMead{Ts <: Simplexer, Tp <: NMParameters} <: ZerothOrderOptimizer
    initial_simplex::Ts
    parameters::Tp
end

"""
# NelderMead
## Constructor
```julia
NelderMead(; parameters = AdaptiveParameters(),
             initial_simplex = AffineSimplexer())
```

The constructor takes 2 keywords:

* `parameters`, an instance of either `AdaptiveParameters` or `FixedParameters`,
and is used to generate parameters for the Nelder-Mead Algorithm
* `initial_simplex`, an instance of `AffineSimplexer`

## Description
Our current implementation of the Nelder-Mead algorithm is based on [1] and [3].
Gradient-free methods can be a bit sensitive to starting values and tuning parameters,
so it is a good idea to be careful with the defaults provided in Optim.jl.

Instead of using gradient information, Nelder-Mead is a direct search method. It keeps
track of the function value at a number of points in the search space. Together, the
points form a simplex. Given a simplex, we can perform one of four actions: reflect,
expand, contract, or shrink. Basically, the goal is to iteratively replace the worst
point with a better point. More information can be found in [1], [2] or [3].

## References
- [1] Nelder, John A. and R. Mead (1965). "A simplex method for function minimization". Computer Journal 7: 308–313. doi:10.1093/comjnl/7.4.308
- [2] Lagarias, Jeffrey C., et al. "Convergence properties of the Nelder–Mead simplex method in low dimensions." SIAM Journal on Optimization 9.1 (1998): 112-147
- [3] Gao, Fuchang and Lixing Han (2010). "Implementing the Nelder-Mead simplex algorithm with adaptive parameters". Computational Optimization and Applications. doi:10.1007/s10589-010-9329-3
"""
function NelderMead(; kwargs...)
    KW = Dict(kwargs)
    if haskey(KW, :initial_simplex) || haskey(KW, :parameters)
        initial_simplex, parameters = AffineSimplexer(), AdaptiveParameters()
        haskey(KW, :initial_simplex) && (initial_simplex = KW[:initial_simplex])
        haskey(KW, :parameters) && (parameters = KW[:parameters])
        return NelderMead(initial_simplex, parameters)
    else
        return NelderMead(AffineSimplexer(), AdaptiveParameters())
    end
end

Base.summary(::NelderMead) = "Nelder-Mead"

# centroid except h-th vertex
function centroid!(c::AbstractArray{T}, simplex, h=0) where T
    n = length(c)
    fill!(c, zero(T))
    @inbounds for i in 1:n+1
        if i != h
            xi = simplex[i]
            c .+= xi
        end
    end
    rmul!(c, T(1)/n)
end

centroid(simplex, h) = centroid!(similar(simplex[1]), simplex, h)

nmobjective(y::Vector, m::Integer, n::Integer) = sqrt(var(y) * (m / n))

function print_header(method::NelderMead)
    @printf "Iter     Function value    √(Σ(yᵢ-ȳ)²)/n \n"
    @printf "------   --------------    --------------\n"
end

function Base.show(io::IO, trace::OptimizationTrace{T, NelderMead}) where T
    @printf io "Iter     Function value    √(Σ(yᵢ-ȳ)²)/n \n"
    @printf io "------   --------------    --------------\n"
    for state in trace.states
        show(io, state)
    end
    return
end

function Base.show(io::IO, t::OptimizationState{NelderMead})
    @printf io "%6d   %14e    %14e\n" t.iteration t.value t.g_norm
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

mutable struct NelderMeadState{Tx, T, Tfs} <: ZerothOrderState
    x::Tx
    m::Int
    simplex::Vector{Tx}
    x_centroid::Tx
    x_lowest::Tx
    x_second_highest::Tx
    x_highest::Tx
    x_reflect::Tx
    x_cache::Tx
    f_simplex::Tfs
    nm_x::T
    f_lowest::T
    i_order::Vector{Int}
    α::T
    β::T
    γ::T
    δ::T
    step_type::String
end

function initial_state(method::NelderMead, options, d, initial_x)
    T = eltype(initial_x)
    n = length(initial_x)
    m = n + 1
    simplex = simplexer(method.initial_simplex, initial_x)
    f_simplex = zeros(T, m)

    value!!(d, first(simplex))

    @inbounds for i in 1:length(simplex)
        f_simplex[i] = value(d, simplex[i])
    end
    # Get the indices that correspond to the ordering of the f values
    # at the vertices. i_order[1] is the index in the simplex of the vertex
    # with the lowest function value, and i_order[end] is the index in the
    # simplex of the vertex with the highest function value
    i_order = sortperm(f_simplex)

    α, β, γ, δ = parameters(method.parameters, n)

NelderMeadState(copy(initial_x), # Variable to hold final minimizer value for MultivariateOptimizationResults
          m, # Number of vertices in the simplex
          simplex, # Maintain simplex in state.simplex
          centroid(simplex,  i_order[m]), # Maintain centroid in state.centroid
          copy(initial_x), # Store cache in state.x_lowest
          copy(initial_x), # Store cache in state.x_second_highest
          copy(initial_x), # Store cache in state.x_highest
          copy(initial_x), # Store cache in state.x_reflect
          copy(initial_x), # Store cache in state.x_cache
          f_simplex, # Store objective values at the vertices in state.f_simplex
          T(nmobjective(f_simplex, n, m)), # Store nmobjective in state.nm_x
          f_simplex[i_order[1]], # Store lowest f in state.f_lowest
          i_order, # Store a vector of rankings of objective values
          T(α),
          T(β),
          T(γ),
          T(δ),
          "initial")
end

function update_state!(f::F, state::NelderMeadState{T}, method::NelderMead) where {F, T}
    # Augment the iteration counter
    shrink = false
    n, m = length(state.x), state.m

    centroid!(state.x_centroid, state.simplex, state.i_order[m])
    copyto!(state.x_lowest, state.simplex[state.i_order[1]])
    copyto!(state.x_second_highest, state.simplex[state.i_order[n]])
    copyto!(state.x_highest, state.simplex[state.i_order[m]])
    state.f_lowest = state.f_simplex[state.i_order[1]]
    f_second_highest = state.f_simplex[state.i_order[n]]
    f_highest = state.f_simplex[state.i_order[m]]

    # Compute a reflection
    @inbounds for j in 1:n
        state.x_reflect[j] = state.x_centroid[j] + state.α * (state.x_centroid[j]-state.x_highest[j])
    end

    f_reflect = value(f, state.x_reflect)
    if f_reflect < state.f_lowest
        # Compute an expansion
        @inbounds for j in 1:n
            state.x_cache[j] = state.x_centroid[j] + state.β *(state.x_reflect[j] - state.x_centroid[j])
        end
        f_expand = value(f, state.x_cache)

        if f_expand < f_reflect
            copyto!(state.simplex[state.i_order[m]], state.x_cache)
            @inbounds state.f_simplex[state.i_order[m]] = f_expand
            state.step_type = "expansion"
        else
            copyto!(state.simplex[state.i_order[m]], state.x_reflect)
            @inbounds state.f_simplex[state.i_order[m]] = f_reflect
            state.step_type = "reflection"
        end
        # shift all order indeces, and wrap the last one around to the first
        i_highest = state.i_order[m]
        @inbounds for i = m:-1:2
            state.i_order[i] = state.i_order[i-1]
        end
        state.i_order[1] = i_highest
    elseif f_reflect < f_second_highest
        copyto!(state.simplex[state.i_order[m]], state.x_reflect)
        @inbounds state.f_simplex[state.i_order[m]] = f_reflect
        state.step_type = "reflection"
        sortperm!(state.i_order, state.f_simplex)
    else
        if f_reflect < f_highest
            # Outside contraction
            @simd for j in 1:n
                @inbounds state.x_cache[j] = state.x_centroid[j] + state.γ * (state.x_reflect[j]-state.x_centroid[j])
            end
            f_outside_contraction = value(f, state.x_cache)
            if f_outside_contraction < f_reflect
                copyto!(state.simplex[state.i_order[m]], state.x_cache)
                @inbounds state.f_simplex[state.i_order[m]] = f_outside_contraction
                state.step_type = "outside contraction"
                sortperm!(state.i_order, state.f_simplex)

            else
                shrink = true
            end
        else # f_reflect > f_highest
            # Inside constraction
            @simd for j in 1:n
                @inbounds state.x_cache[j] = state.x_centroid[j] - state.γ *(state.x_reflect[j] - state.x_centroid[j])
            end
            f_inside_contraction = value(f, state.x_cache)
            if f_inside_contraction < f_highest
                copyto!(state.simplex[state.i_order[m]], state.x_cache)
                @inbounds state.f_simplex[state.i_order[m]] = f_inside_contraction
                state.step_type = "inside contraction"
                sortperm!(state.i_order, state.f_simplex)
            else
                shrink = true
            end
        end
    end

    if shrink
        for i = 2:m
            ord = state.i_order[i]
            copyto!(state.simplex[ord], state.x_lowest + state.δ*(state.simplex[ord]-state.x_lowest))
            state.f_simplex[ord] = value(f, state.simplex[ord])
        end
        step_type = "shrink"
        sortperm!(state.i_order, state.f_simplex)
    end

    state.nm_x = nmobjective(state.f_simplex, n, m)
    false
end

function after_while!(f, state, method::NelderMead, options)
    sortperm!(state.i_order, state.f_simplex)
    x_centroid_min = centroid(state.simplex, state.i_order[state.m])
    f_centroid_min = value(f, x_centroid_min)
    f_min, i_f_min = findmin(state.f_simplex)
    x_min = state.simplex[i_f_min]
    if f_centroid_min < f_min
        x_min = x_centroid_min
        f_min = f_centroid_min
    end
    f.F = f_min
    state.x .= x_min
end
# We don't have an f_x_previous in NelderMeadState, so we need to special case these
pick_best_x(f_increased, state::NelderMeadState) = state.x
pick_best_f(f_increased, state::NelderMeadState, d) = value(d)

function assess_convergence(state::NelderMeadState, d, options::Options)
    g_converged = state.nm_x <= options.g_abstol # Hijact g_converged for NM stopping criterior
    return false, false, g_converged, g_converged, false
end

function initial_convergence(d, state::NelderMeadState, method::NelderMead, initial_x, options)
    nmobjective(state.f_simplex, state.m, length(initial_x)) < options.g_abstol
end

function trace!(tr, d, state, iteration, method::NelderMead, options, curr_time=time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["centroid"] = copy(state.x_centroid)
        dt["step_type"] = state.step_type
    end
    if options.trace_simplex
        dt["simplex"] = state.simplex
        dt["simplex_values"] = state.f_simplex
    end
    update!(tr,
    iteration,
    state.f_lowest,
    state.nm_x,
    dt,
    options.store_trace,
    options.show_trace,
    options.show_every,
    options.callback,
    options.trace_simplex)
end
