centroid(p::Matrix) = reshape(mean(p, 2), size(p, 1))

function dominates(x::Vector, y::Vector)
    for i in 1:length(x)
        @inbounds if x[i] <= y[i]
            return false
        end
    end
    return true
end

function dominates(x::Real, y::Vector)
    for i in 1:length(y)
        @inbounds if x <= y[i]
            return false
        end
    end
    return true
end

nmobjective(y::Vector, m::Integer, n::Integer) = sqrt(var(y) * (m / n))

macro nmtrace()
    quote
        if tracing
            dt = Dict()
            if o.extended_trace
                dt["x"] = copy(x)
            end
            grnorm = NaN
            update!(tr,
                    iteration,
                    f_x,
                    grnorm,
                    dt,
                    o.store_trace,
                    o.show_trace,
                    o.show_every,
                    o.callback)
        end
    end
end

immutable NelderMead <: Optimizer
    a::Float64
    g::Float64
    b::Float64
end

NelderMead(; a::Real = 1.0, g::Real = 2.0, b::Real = 0.5) = NelderMead(a, g, b)

function optimize{T}(f::Function,
                     initial_x::Vector{T},
                     nm::NelderMead,
                     o::OptimizationOptions;
                     initial_step::Vector{T} = ones(T,length(initial_x)))
    # Print header if show_trace is set
    print_header(o)

    # Set up a simplex of points around starting value
    m = length(initial_x)
    if m == 1
        error("Use optimize(f, scalar, scalar) for 1D problems")
    end
    n = m + 1
    p = repmat(initial_x, 1, n)
    @simd for i in 1:m
        @inbounds p[i, i] += initial_step[i]
    end

    # Count function calls
    f_calls = 0

    # Maintain a record of the value of f() at n points
    y = Array(Float64, n)
    for i in 1:n
        @inbounds y[i] = f(p[:, i])
    end
    f_calls += n

    f_x_previous, f_x = NaN, nmobjective(y, m, n)

    # Count iterations
    iteration = 0

    # Maintain a trace
    tr = OptimizationTrace()
    tracing = o.show_trace || o.store_trace || o.extended_trace || o.callback != nothing
    @nmtrace

    # Cache p_bar, y_bar, p_star, p_star_star, p_l, p_h
    p_bar = Array(T, m)
    y_bar = Array(T, m)
    p_star = Array(T, m)
    p_star_star = Array(T, m)
    p_l = Array(T, m)
    p_h = Array(T, m)

    # Iterate until convergence or exhaustion
    f_converged = false
    while !f_converged && iteration < o.iterations
        # Augment the iteration counter
        iteration += 1

        # Find p_l and p_h, the minimum and maximum values of f() among p
        y_l, l = findmin(y)
        copy!(p_l, slice(p, :, l))
        y_h, h = findmax(y)
        copy!(p_h, slice(p, :, h))

        # Compute the centroid of the non-maximal points
        # Also cache function values of all non-maximal points
        fill!(p_bar, 0.0)

        tmpindex = 0
        for i in 1:n
            if i != h
                tmpindex += 1
                @inbounds y_bar[tmpindex] = y[i]
                LinAlg.axpy!(1, slice(p, :, i), p_bar)
            end
        end
        scale!(p_bar, 1/m)

        # Compute a reflection
        @simd for j in 1:m
            @inbounds p_star[j] = (1 + nm.a) * p_bar[j] - nm.a * p_h[j]
        end
        y_star = f(p_star)
        f_calls += 1

        if y_star < y_l
            # Compute an expansion
            @simd for j in 1:m
                @inbounds p_star_star[j] = nm.g * p_star[j] + (1 - nm.g) * p_bar[j]
            end
            y_star_star = f(p_star_star)
            f_calls += 1

            if y_star_star < y_l
                copy!(p_h, p_star_star)
                copy!(slice(p, :, h), p_star_star)
                @inbounds y[h] = y_star_star
            else
                copy!(p_h, p_star)
                copy!(slice(p, :, h), p_star)
                @inbounds y[h] = y_star
            end
        else
            if dominates(y_star, y_bar)
                if y_star < y_h
                    copy!(p_h, p_star)
                    copy!(slice(p, :, h), p_h)
                    @inbounds y[h] = y_star
                end

                # Compute a contraction
                @simd for j in 1:m
                    @inbounds p_star_star[j] = nm.b * p_h[j] + (1 - nm.b) * p_bar[j]
                end
                y_star_star = f(p_star_star)
                f_calls += 1

                if y_star_star > y_h
                    for i = 1:n
                        @simd for j in 1:m
                            @inbounds p[j, i] = (p[j, i] + p_l[j]) / 2.0
                        end
                        @inbounds y[i] = f(p[:, i])
                    end
                else
                    copy!(p_h, p_star_star)
                    copy!(slice(p, :, h), p_h)
                    @inbounds y[h] = y_star_star
                end
            else
                copy!(p_h, p_star)
                copy!(slice(p, :, h), p_h)
                @inbounds y[h] = y_star
            end
        end

        f_x_previous, f_x = f_x, nmobjective(y, m, n)

        @nmtrace

        if f_x <= o.ftol
            f_converged = true
        end
    end

    minimum = centroid(p)

    return MultivariateOptimizationResults("Nelder-Mead",
                                           initial_x,
                                           minimum,
                                           Float64(f(minimum)),
                                           iteration,
                                           iteration == o.iterations,
                                           false,
                                           NaN,
                                           f_converged,
                                           o.ftol,
                                           false,
                                           NaN,
                                           tr,
                                           f_calls,
                                           0)
end
