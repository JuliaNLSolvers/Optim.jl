abstract Simplexer

immutable AffineSimplexer <: Simplexer
    a::Float64
    b::Float64
end
AffineSimplexer(;a = 0.025, b = 0.5) = AffineSimplexer(a, b)

function simplexer{T, N}(S::AffineSimplexer, initial_x::Array{T, N})
    n = length(initial_x)
    initial_simplex = Array{T, N}[copy(initial_x) for i = 1:n+1]
    for j = 1:n
        initial_simplex[j+1][j] = (1+S.b) * initial_simplex[j+1][j] + S.a
    end
    initial_simplex
end

abstract NMParameters

immutable AdaptiveParameters <: NMParameters
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
end

AdaptiveParameters(;  α = 1.0, β = 1.0, γ = 0.75 , δ = 1.0) = AdaptiveParameters(α, β, γ, δ)
parameters(P::AdaptiveParameters, n::Integer) = (P.α, P.β + 2/n, P.γ - 1/2n, P.δ - 1/n)

immutable FixedParameters <: NMParameters
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
end

FixedParameters(; α = 1.0, β = 2.0, γ = 0.5, δ = 0.5) = FixedParameters(α, β, γ, δ)
parameters(P::FixedParameters, n::Integer) = (P.α, P.β, P.γ, P.δ)


immutable NelderMead{Ts <: Simplexer, Tp <: NMParameters} <: Optimizer
    initial_simplex::Ts
    parameters::Tp
end

method_string(method::NelderMead) = "Nelder-Mead"

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
function centroid!{T}(c::Array{T}, simplex, h=0)
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

function Base.show(io::IO, trace::OptimizationTrace{NelderMead})
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

type NelderMeadState{T, N}
    m::Int64
    n::Int64
    simplex::Vector{Array{T,N}}
    x_centroid::Array{T}
    x::Array{T}
    x_lowest::Array{T}
    x_second_highest::Array{T}
    x_highest::Array{T}
    x_reflect::Array{T}
    x_cache::Array{T}
    f_simplex::Array{T}
    f_x_previous::T
    f_x::T
    f_lowest::T
    i_order::Vector{Int64}
    α::T
    β::T
    γ::T
    δ::T
    step_type::String
    f_calls::Int64
    g_calls::Int64
end

initialize_state(method::NelderMead, options, d, initial_x::Array) = initialize_state(method, options, d.f, initial_x)

function initialize_state{T}(method::NelderMead, options, f::Function, initial_x::Array{T})
    m = length(initial_x)
    n = m + 1
    simplex = simplexer(method.initial_simplex, initial_x)
    f_simplex = zeros(T, n)
    @inbounds for i in 1:length(simplex)
        f_simplex[i] = f(simplex[i])
    end
    # Get the indeces that correspond to the ordering of the f values
    # at the vertices. i_order[1] is the index in the simplex of the vertex
    # with the lowest function value, and i_order[end] is the index in the
    # simplex of the vertex with the highest function value
    i_order = sortperm(f_simplex)

    α, β, γ, δ = parameters(method.parameters, m)

NelderMeadState(m, # Dimensionality of the problem
          n, # Number of vertices in the simplex
          simplex, # Maintain simplex in state.simplex
          centroid(simplex,  i_order[n]), # Maintain centroid in state.centroid
          Array{T}(m), # Variable to hold final minimizer value for MultivariateOptimizationResults
          Array{T}(m), # Store cache in state.x_lowest
          Array{T}(m), # Store cache in state.x_second_highest
          Array{T}(m), # Store cache in state.x_highest
          Array{T}(m), # Store cache in state.x_reflect
          Array{T}(m), # Store cache in state.x_cache
          f_simplex, # Store objective values at the vertices in state.f_simplex
          T(NaN), # Store previous f in state.f_x_previous
          T(nmobjective(f_simplex, m, n)), # Store Nelder Mead objective in state.f_x
          f_simplex[i_order[1]], # Store lowest f in state.f_lowest
          i_order, # Store a vector of rankings of objective values
          T(α),
          T(β),
          T(γ),
          T(δ),
          "initial",
          n,
          0) # Track f calls in state.f_calls
end
update!(d, state::NelderMeadState, method::NelderMead) = update!(d.f, state, method)
function update!{T}(f::Function, state::NelderMeadState{T}, method::NelderMead)
    # Augment the iteration counter
    shrink = false
    m, n = state.m, state.n
    centroid!(state.x_centroid, state.simplex,  state.i_order[n])
    copy!(state.x_lowest, state.simplex[state.i_order[1]])
    copy!(state.x_second_highest, state.simplex[state.i_order[m]])
    copy!(state.x_highest, state.simplex[state.i_order[n]])

    f_lowest = state.f_simplex[state.i_order[1]]
    f_second_highest = state.f_simplex[state.i_order[m]]
    f_highest = state.f_simplex[state.i_order[n]]
    # Compute a reflection
    @inbounds for j in 1:m
        state.x_reflect[j] = state.x_centroid[j] + state.α * (state.x_centroid[j]-state.x_highest[j])
    end

    f_reflect = f(state.x_reflect)
    state.f_calls += 1
    if f_reflect < f_lowest
        # Compute an expansion
        @inbounds for j in 1:m
            state.x_cache[j] = state.x_centroid[j] + state.β *(state.x_reflect[j] - state.x_centroid[j])
        end
        f_expand = f(state.x_cache)
        state.f_calls += 1

        if f_expand < f_reflect
            copy!(state.simplex[state.i_order[n]], state.x_cache)
            @inbounds state.f_simplex[state.i_order[n]] = f_expand
            state.step_type = "expansion"
        else
            copy!(state.simplex[state.i_order[n]], state.x_reflect)
            @inbounds state.f_simplex[state.i_order[n]] = f_reflect
            state.step_type = "reflection"
        end
        # shift all order indeces, and wrap the last one around to the first
        i_highest = state.i_order[n]
        @inbounds for i = n:-1:2
            state.i_order[i] = state.i_order[i-1]
        end
        state.i_order[1] = i_highest
    elseif f_reflect < f_second_highest
        copy!(state.simplex[state.i_order[n]], state.x_reflect)
        @inbounds state.f_simplex[state.i_order[n]] = f_reflect
        state.step_type = "reflection"
        sortperm!(state.i_order, state.f_simplex)
    else
        if f_reflect < f_highest
            # Outside contraction
            @simd for j in 1:m
                @inbounds state.x_cache[j] = state.x_centroid[j] + state.γ * (state.x_reflect[j]-state.x_centroid[j])
            end
            f_outside_contraction = f(state.x_cache)
            if f_outside_contraction < f_reflect
                copy!(state.simplex[state.i_order[n]], state.x_cache)
                @inbounds state.f_simplex[state.i_order[n]] = f_outside_contraction
                state.step_type = "outside contraction"
                sortperm!(state.i_order, state.f_simplex)

            else
                shrink = true
            end
        else # f_reflect > f_highest
            # Inside constraction
            @simd for j in 1:m
                @inbounds state.x_cache[j] = state.x_centroid[j] - γ *(state.x_reflect[j] - state.x_centroid[j])
            end
            f_inside_contraction = f(state.x_cache)
            if f_inside_contraction < f_highest
                copy!(state.simplex[ state.i_order[n]], state.x_cache)
                @inbounds state.f_simplex[ state.i_order[n]] = f_inside_contraction
                state.step_type = "inside contraction"
                sortperm!(state.i_order, state.f_simplex)
            else
                shrink = true
            end
        end
    end

    if shrink
        for i = 2:n
            ord = i_order[i]
            copy!(simplex[ord], x_lowest + δ*(simplex[ord]-x_lowest))
            f_simplex[ord] = f(simplex[ord])
        end
        step_type = "shrink"
        sortperm!(i_order, f_simplex)
    end

    state.f_x_previous, state.f_x = state.f_x, nmobjective(state.f_simplex, m, n)
    false
end

after_while!(d, state, method::NelderMead, options) = after_while!(d.f, state, method::NelderMead, options)
function after_while!(f::Function, state, method::NelderMead, options)
    sortperm!(state.i_order, state.f_simplex)
    x_centroid_min = centroid(state.simplex,  state.i_order[state.n])
    f_centroid_min = f(state.x_centroid)
    state.f_calls += 1
    f_min, i_f_min = findmin(state.f_simplex)
    x_min = state.simplex[i_f_min]
    if f_centroid_min < f_min
        x_min = x_centroid_min
        f_min = f_centroid_min
    end
    state.f_x = f_min
    state.x[:] = x_min
end
