# TODO: Avoid recomputing centroids and f(centroid)
function nelder_mead_trace!(tr::OptimizationTrace,
                            p::Matrix,
                            y::Vector,
                            n::Integer,
                            f::Function,
                            iteration::Integer,
                            store_trace::Bool,
                            show_trace::Bool)
    d = Dict()
    d["Standard Deviation over Simplex"] = std(y)
    os = OptimizationState(centroid(p), f(centroid(p)), iteration, d)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function nelder_mead(f::Function,
                     initial_x::Vector;
                     a::Real = 1.0,
                     g::Real = 2.0,
                     b::Real = 0.5,
                     tolerance::Real = 1e-8,
                     iterations::Integer = 1_000,
                     store_trace::Bool = false,
                     show_trace::Bool = false)

    # Set up a simplex of points around starting value
    m = length(initial_x)
    n = m + 1
    p = repmat(initial_x, 1, n)
    for i in 1:length(p)
        p[i] += randn()
    end

    # Count function calls
    f_calls = 0

    # Maintain a record of the value of f() at n points
    y = Array(Float64, n)
    for i in 1:n
        y[i] = f(p[:, i])
    end
    f_calls += n

    # Count iterations
    iteration = 0

    # Maintain a trace
    tr = OptimizationTrace()
    if store_trace || show_trace
        nelder_mead_trace!(tr,
                           p,
                           y,
                           n,
                           f,
                           iteration,
                           store_trace,
                           show_trace)
    end

    # Monitor convergence
    converged = false

    # Cache p_bar, y_bar, p_star and p_star_star
    p_bar = Array(Float64, m)
    y_bar = Array(Float64, m)
    p_star = Array(Float64, m)
    p_star_star = Array(Float64, m)

    # Iterate until convergence or exhaustion
    while !converged && iteration < iterations
        # Augment the iteration counter
        iteration += 1

        # Find p_l and p_h, the minimum and maximum values of f() among p
        y_l, l = findmin(y)
        p_l = p[:, l]
        y_h, h = findmax(y)
        p_h = p[:, h]

        # Compute the centroid of the non-maximal points
        # Also cache function values of all non-maximal points
        fill!(p_bar, 0.0)
        let
            tmpindex = 0
            for i in 1:n
                if i != h
                    tmpindex += 1
                    y_bar[tmpindex] = y[i]
                    p_bar[:] += p[:, i]
                end
            end
        end
        for j in 1:m
            p_bar[j] /= m
        end

        # Compute a reflection
        for j in 1:m
            p_star[j] = (1 + a) * p_bar[j] - a * p_h[j]
        end
        y_star = f(p_star)
        f_calls += 1

        if y_star < y_l
            # Compute an expansion
            for j in 1:m
                p_star_star[j] = g * p_star[j] + (1 - g) * p_bar[j]
            end
            y_star_star = f(p_star_star)
            f_calls += 1

            if y_star_star < y_l
                p_h[:] = p_star_star
                p[:, h] = p_star_star
                y[h] = y_star_star
            else
                p_h = p_star
                p[:, h] = p_star
                y[h] = y_star
            end
        else
            if all(y_star .> y_bar)
                if y_star < y_h
                    p_h[:] = p_star
                    p[:, h] = p_h
                    y[h] = y_star
                end

                # Compute a contraction
                for j in 1:m
                    p_star_star[j] = b * p_h[j] + (1 - b) * p_bar[j]
                end
                y_star_star = f(p_star_star)
                f_calls += 1

                if y_star_star > y_h
                    for i = 1:n
                        for j in 1:m
                            p[j, i] = (p[j, i] + p_l[j]) / 2.0
                        end
                        y[i] = f(p[:, i])
                    end
                else
                    p_h[:] = p_star_star
                    p[:, h] = p_h
                    y[h] = y_star_star
                end
            else
                p_h[:] = p_star
                p[:, h] = p_h
                y[h] = y_star
            end
        end

        v = sqrt(var(y) * (m / n))

        if store_trace || show_trace
            nelder_mead_trace!(tr,
                               p,
                               y,
                               n,
                               f,
                               iteration,
                               store_trace,
                               show_trace)
        end

        if v <= tolerance
            converged = true
        end
    end

    OptimizationResults("Nelder-Mead",
                        initial_x,
                        centroid(p),
                        f(centroid(p)),
                        iteration,
                        converged,
                        tr,
                        f_calls,
                        0)
end
