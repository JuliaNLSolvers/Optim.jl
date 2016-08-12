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

macro nmtrace()
    quote
        if tracing
            dt = Dict()
            if o.extended_trace
                dt["centroid"] = x_centroid
                dt["step_type"] = step_type
            end
            update!(tr,
                    iteration,
                    f_lowest,
                    f_x,
                    dt,
                    o.store_trace,
                    o.show_trace,
                    o.show_every,
                    o.callback)
        end
    end
end

function print_header(mo::NelderMead, options::OptimizationOptions)
    if options.show_trace
        @printf "Iter     Function value    √(Σ(yᵢ-ȳ)²)/n \n"
        @printf "------   --------------    --------------\n"
    end
end

function Base.show(io::IO, t::OptimizationTrace{NelderMead})
    @printf io "Iter     Function value    √(Σ(yᵢ-ȳ)²)/n \n"
    @printf io "------   --------------    --------------\n"
    for state in t.states
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

function optimize{T}(f::Function,
                     initial_x::Vector{T},
                     mo::NelderMead,
                     o::OptimizationOptions)

    # Print header if show_trace is set
    print_header(mo, o)

    # Set up a simplex of points
    m = length(initial_x)
    if m == 1
        error("Use optimize(f, scalar, scalar) for 1D problems")
    end
    n = m + 1
    simplex = simplexer(mo.initial_simplex, initial_x)
    f_simplex = zeros(T, n)
    @inbounds for i in 1:length(simplex)
        f_simplex[i] = f(simplex[i])
    end

    # Get the indeces that correspond to the ordering of the f values
    # at the vertices. i_order[1] is the index in the simplex of the vertex
    # with the lowest function value, and i_order[end] is the index in the
    # simplex of the vertex with the highest function value
    i_order = sortperm(f_simplex)
    x_centroid = centroid(simplex,  i_order[n])

    # Count function calls
    f_calls = n

    # Setup parameters
    α, β, γ, δ = parameters(mo.parameters, m)
    # Count iterations
    iteration = 0

    step_type = "initial"

    # Maintain a trace
    f_x_previous, f_x = NaN, nmobjective(f_simplex, m, n)
    f_lowest = f_simplex[i_order[1]]
    tr = OptimizationTrace{typeof(mo)}()
    tracing = o.show_trace || o.store_trace || o.extended_trace || o.callback != nothing
    @nmtrace

    # Cache x_centroid, y_bar, x_reflect, x_expand, x_lowest, x_highest
    x_reflect = Array(T, m)
    x_expand = Array(T, m)
    x_cache = Array(T, m)

    x_lowest = Array(T, m)
    x_second_highest = Array(T, m)
    x_highest = Array(T, m)

    # Iterate until convergence or exhaustion
    x_converged = false
    f_converged = false
    g_converged = false

    while !g_converged && !f_converged && iteration < o.iterations
        # Augment the iteration counter
        shrink = false
        iteration += 1

        centroid!(x_centroid, simplex,  i_order[n])
        copy!(x_lowest, simplex[i_order[1]])
        copy!(x_second_highest, simplex[i_order[m]])
        copy!(x_highest, simplex[ i_order[n]])

        f_lowest = f_simplex[i_order[1]]
        f_second_highest = f_simplex[i_order[m]]
        f_highest = f_simplex[ i_order[n]]
        # Compute a reflection
        @inbounds for j in 1:m
            x_reflect[j] = x_centroid[j] + α * (x_centroid[j]-x_highest[j])
        end

        f_reflect = f(x_reflect)
        f_calls += 1
        if f_reflect < f_lowest
            # Compute an expansion
            @inbounds for j in 1:m
                x_cache[j] = x_centroid[j] + β *(x_reflect[j] - x_centroid[j])
            end
            f_expand = f(x_cache)
            f_calls += 1

            if f_expand < f_reflect
                copy!(simplex[ i_order[n]], x_cache)
                @inbounds f_simplex[ i_order[n]] = f_expand
                step_type = "expansion"
            else
                copy!(simplex[ i_order[n]], x_reflect)
                @inbounds f_simplex[ i_order[n]] = f_reflect
                step_type = "reflection"
            end
            # shift all order indeces, and wrap the last one around to the first
            i_highest = i_order[n]
            @inbounds for i = n:-1:2
                i_order[i] = i_order[i-1]
            end
            i_order[1] = i_highest
        elseif f_reflect < f_second_highest
            copy!(simplex[ i_order[n]], x_reflect)
            @inbounds f_simplex[ i_order[n]] = f_reflect
            step_type = "reflection"
            sortperm!(i_order, f_simplex)
        else
            if f_reflect < f_highest
                # Outside contraction
                @simd for j in 1:m
                    @inbounds x_cache[j] = x_centroid[j] + γ * (x_reflect[j]-x_centroid[j])
                end
                f_outside_contraction = f(x_cache)
                if f_outside_contraction < f_reflect
                    copy!(simplex[ i_order[n]], x_cache)
                    @inbounds f_simplex[ i_order[n]] = f_outside_contraction
                    step_type = "outside contraction"
                    sortperm!(i_order, f_simplex)

                else
                    shrink = true
                end
            else # f_reflect > f_highest
                # Inside constraction
                @simd for j in 1:m
                    @inbounds x_cache[j] = x_centroid[j] - γ *(x_reflect[j] - x_centroid[j])
                end
                f_inside_contraction = f(x_cache)
                if f_inside_contraction < f_highest
                    copy!(simplex[ i_order[n]], x_cache)
                    @inbounds f_simplex[ i_order[n]] = f_inside_contraction
                    step_type = "inside contraction"
                    sortperm!(i_order, f_simplex)
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

        f_x_previous, f_x = f_x, nmobjective(f_simplex, m, n)

        @nmtrace
        if f_x <= o.g_tol
            g_converged = true
        end
    end

    sortperm!(i_order, f_simplex)
    x_centroid_min = centroid(simplex,  i_order[n])
    f_centroid_min = f(x_centroid)
    f_calls += 1
    f_min, i_f_min = findmin(f_simplex)
    x_min = simplex[i_f_min]
    if f_centroid_min < f_min
        x_min = x_centroid_min
        f_min = f_centroid_min
    end

    return MultivariateOptimizationResults("Nelder-Mead",
                                           initial_x,
                                           x_min,
                                           Float64(f_min),
                                           iteration,
                                           iteration == o.iterations,
                                           x_converged,
                                           NaN,
                                           f_converged,
                                           NaN,
                                           g_converged,
                                           o.g_tol,
                                           tr,
                                           f_calls,
                                           0)
end
