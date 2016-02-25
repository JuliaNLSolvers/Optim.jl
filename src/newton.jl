macro newtontrace()
    quote
        if tracing
            dt = Dict()
            if o.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(gr)
                dt["h(x)"] = copy(H)
            end
            grnorm = norm(gr, Inf)
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

immutable Newton <: Optimizer
    linesearch!::Function
end

Newton(; linesearch!::Function = hz_linesearch!) =
  Newton(linesearch!)

function optimize(d::Function,
                  initial_x::Array,
                  method::Newton,
                  options::OptimizationOptions)
    throw(ArgumentError("Newton's method without a user supplied gradient and hessian is currently not supported."))
end

function optimize{T}(d::TwiceDifferentiableFunction,
                     initial_x::Vector{T},
                     mo::Newton,
                     o::OptimizationOptions)
    # Print header if show_trace is set
    print_header(o)

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy(initial_x), copy(initial_x)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls, g_calls = 0, 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in gr
    gr = Array(T, n)

    # The current search direction
    # TODO: Try to avoid re-allocating s
    s = Array(T, n)

    # Buffers for use in line search
    x_ls, gr_ls = Array(T, n), Array(T, n)

    # Store f(x) in f_x
    f_x_previous, f_x = NaN, d.fg!(x, gr)
    f_calls, g_calls = f_calls + 1, g_calls + 1

    # Store h(x) in H
    H = Array(T, n, n)
    d.h!(x, H)

    # Keep track of step-sizes
    alpha = alphainit(one(T), x, gr, f_x)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = o.store_trace || o.show_trace || o.extended_trace || o.callback != nothing
    @newtontrace

    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = false, false, false

    # Iterate until convergence
    converged = false
    while !converged && iteration < o.iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Search direction is always the negative gradient divided by H
        # TODO: Do this calculation in place
        @inbounds s[:] = -(H \ gr)

        # Refresh the line search cache
        dphi0 = vecdot(gr, s)
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          mo.linesearch!(d, x, s, x_ls, gr_ls, lsr, alpha, mayterminate)
        f_calls, g_calls = f_calls + f_update, g_calls + g_update

        # Maintain a record of previous position
        copy!(x_previous, x)

        # Update current position # x = x + alpha * s
        LinAlg.axpy!(alpha, s, x)

        # Update the function value and gradient
        f_x_previous, f_x = f_x, d.fg!(x, gr)
        f_calls, g_calls = f_calls + 1, g_calls + 1

        # Update the Hessian
        d.h!(x, H)

        x_converged,
        f_converged,
        gr_converged,
        converged = assess_convergence(x,
                                       x_previous,
                                       f_x,
                                       f_x_previous,
                                       gr,
                                       o.xtol,
                                       o.ftol,
                                       o.grtol)

        @newtontrace
    end

    return MultivariateOptimizationResults("Newton's Method",
                                           initial_x,
                                           x,
                                           Float64(f_x),
                                           iteration,
                                           iteration == o.iterations,
                                           x_converged,
                                           o.xtol,
                                           f_converged,
                                           o.ftol,
                                           gr_converged,
                                           o.grtol,
                                           tr,
                                           f_calls,
                                           g_calls)
end
