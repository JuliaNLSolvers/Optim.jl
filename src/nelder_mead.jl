##############################################################################
##
## REF: http://optlab-server.sce.carleton.ca/POAnimations2007/NonLinear7.html
##
## Initial points
## N + 1 points of dimension N
## Store in N + 1 by N array
## Compute centroid of non-maximal points
## Iteratively update p_h
## Parameters: a, g, b
## y_l is current minimum
## y_h is current maximum
##
## TODO: Switch over to column-major form for points.
##
##############################################################################

function nelder_mead(f::Function,
                     initial_p::Matrix,
                     a::Float64,
                     g::Float64,
                     b::Float64,
                     tolerance::Float64,
                     max_iterations::Int64,
                     store_trace::Bool,
                     show_trace::Bool)

    # Center the algorithm around an arbitrary point.
    p = copy(initial_p)

    # Confirm that we have a proper simplex.
    m = size(p, 1)
    n = size(p, 2)
    if m != n - 1
        error("A simplex in n-dimensions must include n + 1 points")
    end

    # Maintain a record of the value of f() at n points.
    y = zeros(n)
    for i = 1:n
        y[i] = f(p[:, i])
    end

    # Don't run forever.
    iter = 0

    tr = OptimizationTrace()

    # Maintain a trace.
    if store_trace || show_trace
        d = Dict()
        v = sqrt(var(y) * ((n - 1) / n))
        d["Variance"] = v
        os = OptimizationState(centroid(p), f(centroid(p)), iter, d)
        if store_trace
            push!(tr, os)
        end
        if show_trace
            println(os)
        end
    end

    # Track convergence.
    converged = false

    # Iterate until convergence or exhaustion.
    while !converged && iter < max_iterations
        # Keep a record of which p's are modified to save on updating y's.
        modified_indices = Array(Int, 0)

        # Augment the iteration counter.
        iter = iter + 1

        # Find p_l and p_h, the minimum and maximum values of f() among p.
        # Always take the first min or max if many exist.
        l = findfirst(y .== min(y))
        p_l = p[:, l]
        y_l = y[l]
        h = findfirst(y .== max(y))
        p_h = p[:, h]
        y_h = y[h]

        # Remove the maximum valued point from our set.
        non_maximal_p = p[:, find([1:n] .!= h)]

        # Compute the centroid of the remaining points.
        p_bar = centroid(non_maximal_p)

        # For later, prestore function values of all non-maximal points.
        y_bar = y[find([1:n] .!= h)]

        # Compute a reflection.
        p_star = (1 + a) * p_bar - a * p_h
        y_star = f(p_star)

        if y_star < y_l
            # Compute an expansion.
            p_star_star = g * p_star + (1 - g) * p_bar
            y_star_star = f(p_star_star)

            if y_star_star < y_l
                p_h = p_star_star
                p[:, h] = p_h
                y[h] = y_star_star
            else
                p_h = p_star
                p[:, h] = p_h
                y[h] = y_star
            end
        else
            if all(y_star .> y_bar)
                if y_star > y_h
                    1 # Do a NO-OP.
                else
                    p_h = p_star
                    p[:, h] = p_h
                    y[h] = y_star
                end

                # Compute a contraction.
                p_star_star = b * p_h + (1 - b) * p_bar
                y_star_star = f(p_star_star)

                if y_star_star > y_h
                    for i = 1:n
                        # This makes reuse tricky.
                        p[:, i] = (p[:, i] + p_l) / 2
                        y[i] = f(p[:, i])
                    end
                else
                    p_h = p_star_star
                    p[:, h] = p_h
                    y[h] = y_star_star
                end
            else
                p_h = p_star
                p[:, h] = p_h
                y[h] = y_star
            end
        end

        v = sqrt(var(y) * ((n - 1) / n))

        if store_trace || show_trace
            d = Dict()
            d["Variance"] = v
            os = OptimizationState(centroid(p), f(centroid(p)), iter, d)
            if store_trace
                push!(tr, os)
            end
            if show_trace
                println(os)
            end
        end

        if v <= tolerance
            converged = true
        end
    end

    OptimizationResults("Nelder-Mead",
                        centroid(initial_p),
                        centroid(p),
                        f(centroid(p)),
                        iter,
                        converged,
                        tr)
end

function nelder_mead(f::Function,
                     initial_p::Matrix)
  nelder_mead(f, initial_p, 1.0, 2.0, 0.5, 10e-8, 1_000, false, false)
end

function nelder_mead(f::Function,
                     initial_x::Vector)
  n = length(initial_x)
  initial_p = hcat(diagm(ones(n)), initial_x)
  nelder_mead(f, initial_p, 1.0, 2.0, 0.5, 10e-8, 1_000, false, false)
end
