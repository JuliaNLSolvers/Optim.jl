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

Base.summary(::NelderMead) = "Nelder-Mead"

function NelderMead(; kwargs...)
    KW = Dict(kwargs)
    if haskey(KW, :a) || haskey(KW, :g) || haskey(KW, :b)
        a, g, b = 1.0, 2.0, 0.5
        haskey(KW, :a) && (a = KW[:a])
        haskey(KW, :g) && (g = KW[:g])
        haskey(KW, :b) && (b = KW[:b])
        return NelderMead(a, g, b)
    elseif haskey(KW, :initial_simplex) || haskey(KW, :parameters)
        initial_simplex, parameters = AffineSimplexer(), AdaptiveParameters()
        haskey(KW, :initial_simplex) && (initial_simplex = KW[:initial_simplex])
        haskey(KW, :parameters) && (parameters = KW[:parameters])
        return NelderMead(initial_simplex, parameters)
    else
        return NelderMead(AffineSimplexer(), AdaptiveParameters())
    end
end

# centroid except h-th vertex
function centroid!(c::AbstractArray{T}, simplex, h=0) where T
    n = length(c)
    fill!(c, zero(T))
    @inbounds for i in 1:n+1
        if i != h
            xi = simplex[i]
            @simd for j in 1:n
                c[j] += xi[j]
            end
        end
    end
    scale!(c, 1/n)
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
    # Get the indeces that correspond to the ordering of the f values
    # at the vertices. i_order[1] is the index in the simplex of the vertex
    # with the lowest function value, and i_order[end] is the index in the
    # simplex of the vertex with the highest function value
    i_order = sortperm(f_simplex)

    α, β, γ, δ = parameters(method.parameters, n)

NelderMeadState(similar(initial_x), # Variable to hold final minimizer value for MultivariateOptimizationResults
          m, # Number of vertices in the simplex
          simplex, # Maintain simplex in state.simplex
          centroid(simplex,  i_order[m]), # Maintain centroid in state.centroid
          similar(initial_x), # Store cache in state.x_lowest
          similar(initial_x), # Store cache in state.x_second_highest
          similar(initial_x), # Store cache in state.x_highest
          similar(initial_x), # Store cache in state.x_reflect
          similar(initial_x), # Store cache in state.x_cache
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
    copy!(state.x_lowest, state.simplex[state.i_order[1]])
    copy!(state.x_second_highest, state.simplex[state.i_order[n]])
    copy!(state.x_highest, state.simplex[state.i_order[m]])
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
            copy!(state.simplex[state.i_order[m]], state.x_cache)
            @inbounds state.f_simplex[state.i_order[m]] = f_expand
            state.step_type = "expansion"
        else
            copy!(state.simplex[state.i_order[m]], state.x_reflect)
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
        copy!(state.simplex[state.i_order[m]], state.x_reflect)
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
                copy!(state.simplex[state.i_order[m]], state.x_cache)
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
                copy!(state.simplex[state.i_order[m]], state.x_cache)
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
            copy!(state.simplex[ord], state.x_lowest + state.δ*(state.simplex[ord]-state.x_lowest))
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
    state.x[:] = x_min
end
# We don't have an f_x_previous in NelderMeadState, so we need to special case these
pick_best_x(f_increased, state::NelderMeadState) = state.x
pick_best_f(f_increased, state::NelderMeadState, d) = value(d)

function assess_convergence(state::NelderMeadState, d, options)
    g_converged = state.nm_x <= options.g_tol # Hijact g_converged for NM stopping criterior
    return false, false, g_converged, g_converged, false
end

function trace!(tr, d, state, iteration, method::NelderMead, options)
    dt = Dict()
    if options.extended_trace
        dt["centroid"] = copy(state.x_centroid)
        dt["step_type"] = state.step_type
    end
    update!(tr,
    iteration,
    state.f_lowest,
    state.nm_x,
    dt,
    options.store_trace,
    options.show_trace,
    options.show_every,
    options.callback)
end
