function gradient_descent_trace!(tr::OptimizationTrace,
                                 x::Vector,
                                 f_x::Real,
                                 gr::Vector,
                                 iteration::Integer,
                                 store_trace::Bool,
                                 show_trace::Bool)
    dt = Dict()
    dt["g(x)"] = copy(gr)
    dt["Maximum component of g(x)"] = norm(gr, Inf)
    os = OptimizationState(copy(x), f_x, iteration, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function gradient_descent{T}(d::Union(DifferentiableFunction,
                                      TwiceDifferentiableFunction),
                             initial_x::Vector{T};
                             tolerance::Real = 1e-8,
                             iterations::Integer = 1_000,
                             store_trace::Bool = false,
                             show_trace::Bool = false,
                             linesearch!::Function = hz_linesearch!)

    # Maintain current state in x
    x = copy(initial_x)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls = 0
    g_calls = 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in gr
    gr = Array(Float64, n)

    # The current search direction
    s = Array(Float64, n)

    # Buffers for use in line search
    x_ls = Array(Float64, n)
    gr_ls = Array(Float64, n)

    # Store f(x) in f_x
    f_x = d.fg!(x, gr)
    f_calls += 1
    g_calls += 1

    # Keep track of step-sizes
    alpha = 1.0
    # TODO: Restore these pieces
    # alpha = alphainit(1.0, x, g???, phi0???)
    # alphamax = Inf # alphamaxfunc(x, d)
    # alpha = min(alphamax, alpha)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace
    if tracing
        gradient_descent_trace!(tr, x, f_x, gr,
                                iteration, store_trace, show_trace)
    end

    # Iterate until convergence
    converged = false
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Search direction is always the negative gradient
        for i in 1:n
            s[i] = -gr[i]
        end

        # Refresh the line search cache
        dphi0 = dot(gr, s)
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        # TODO: Fix f_update, g_update
        alpha, f_update, g_update =
          linesearch!(d, x, s, x_ls, gr_ls, lsr, alpha, mayterminate)
        f_calls += f_update
        g_calls += g_update

        # Update current position
        for i in 1:n
            x[i] = x[i] + alpha * s[i]
        end

        # Update the function value and gradient
        f_x = d.fg!(x, gr)
        f_calls += 1
        g_calls += 1

        # Assess convergence
        if norm(gr, Inf) <= tolerance
            converged = true
        end

        # Show trace
        if tracing
            gradient_descent_trace!(tr, x, f_x, gr,
                                    iteration, store_trace, show_trace)
        end
    end

    OptimizationResults("Gradient Descent",
                        initial_x,
                        x,
                        f_x,
                        iteration,
                        converged,
                        tr,
                        f_calls,
                        g_calls)
end
