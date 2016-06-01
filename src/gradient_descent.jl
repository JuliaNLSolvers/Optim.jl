

macro gdtrace()
    quote
        if tracing
            dt = Dict()
            if o.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(g)
            end
            g_norm = vecnorm(g, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    g_norm,
                    dt,
                    o.store_trace,
                    o.show_trace,
                    o.show_every,
                    o.callback)
        end
    end
end

immutable GradientDescent{T} <: Optimizer
    linesearch!::Function
    P::T
    precondprep!::Function
end

GradientDescent(; linesearch!::Function = hz_linesearch!,
                P = nothing, precondprep! = (P, x) -> nothing) =
                    GradientDescent(linesearch!, P, precondprep!)

function optimize{T}(d::DifferentiableFunction,
                     initial_x::Array{T},
                     mo::GradientDescent,
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

    # Maintain current gradient in g
    g = similar(x)

    # The current search direction
    s = similar(x)

    # Buffers for use in line search
    x_ls = similar(x)
    g_ls = similar(x)

    # Store f(x) in f_x
    f_x_previous, f_x = NaN, d.fg!(x, g)
    f_calls, g_calls = f_calls + 1, g_calls + 1

    # Keep track of step-sizes
    alpha = alphainit(one(T), x, g, f_x)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Trace the history of states visited
    tr = OptimizationTrace(mo)
    tracing = o.store_trace || o.show_trace || o.extended_trace || o.callback != nothing
    @gdtrace

    # Assess multiple types of convergence
    x_converged, f_converged, g_converged = false, false, false

    # Iterate until convergence
    converged = false
    while !converged && iteration < o.iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Search direction is always the negative preconditioned gradient
        mo.precondprep!(mo.P, x)
        A_ldiv_B!(s, mo.P, g)
        @simd for i in 1:n
            @inbounds s[i] = -s[i]
        end

        # Refresh the line search cache
        dphi0 = vecdot(g, s)
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          mo.linesearch!(d, x, s, x_ls, g_ls, lsr, alpha, mayterminate)
        f_calls, g_calls = f_calls + f_update, g_calls + g_update

        # Maintain a record of previous position
        copy!(x_previous, x)

        # Update current position # x = x + alpha * s
        LinAlg.axpy!(alpha, s, x)

        # Update the function value and gradient
        f_x_previous, f_x = f_x, d.fg!(x, g)
        f_calls, g_calls = f_calls + 1, g_calls + 1

        x_converged,
        f_converged,
        g_converged,
        converged = assess_convergence(x,
                                       x_previous,
                                       f_x,
                                       f_x_previous,
                                       g,
                                       o.x_tol,
                                       o.f_tol,
                                       o.g_tol)

        @gdtrace
    end

    return MultivariateOptimizationResults("Gradient Descent",
                                           initial_x,
                                           x,
                                           Float64(f_x),
                                           iteration,
                                           iteration == o.iterations,
                                           x_converged,
                                           o.x_tol,
                                           f_converged,
                                           o.f_tol,
                                           g_converged,
                                           o.g_tol,
                                           tr,
                                           f_calls,
                                           g_calls)
end
