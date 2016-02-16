function backtracking_linesearch!{T}(d::Union{DifferentiableFunction,
                                              TwiceDifferentiableFunction},
                                     x::Vector{T},
                                     s::Vector,
                                     x_scratch::Vector,
                                     gr_scratch::Vector,
                                     lsr::LineSearchResults,
                                     alpha::Real = 1.0,
                                     mayterminate::Bool = false,
                                     c1::Real = 1e-4,
                                     c2::Real = 0.9,
                                     rho::Real = 0.9,
                                     iterations::Integer = 1_000)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls = 0
    g_calls = 0

    # Count number of parameters
    n = length(x)

    # Store f(x) in f_x
    f_x = d.fg!(x, gr_scratch)
    f_calls += 1
    g_calls += 1

    # Store angle between search direction and gradient
    gxp = vecdot(gr_scratch, s)

    # Tentatively move a distance of alpha in the direction of s
    @simd for i in 1:n
        @inbounds x_scratch[i] = x[i] + alpha * s[i]
    end

    # Backtrack until we satisfy sufficient decrease condition
    f_x_scratch = d.f(x_scratch)
    f_calls += 1
    while f_x_scratch > f_x + c1 * alpha * gxp
        # Increment the number of steps we've had to perform
        iteration += 1

        # Ensure termination
        if iteration > iterations
            error("Too many iterations in backtracking_linesearch!")
        end

        # Shrink proposed step-size
        alpha *= rho

        # Update proposed position
        @simd for i in 1:n
            @inbounds x_scratch[i] = x[i] + alpha * s[i]
        end

        # Evaluate f(x) at proposed position
        f_x_scratch = d.f(x_scratch)
        f_calls += 1
    end

    return alpha, f_calls, g_calls
end
