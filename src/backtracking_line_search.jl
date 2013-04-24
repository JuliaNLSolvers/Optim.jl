# From Andreas' changes and Nocedal & Wright
# c1::Real = 1e-6
# c2::Real = 0.9
# rho::Real = 0.9

# From Boyd and Vanderberghe

function backtracking_line_search!(d::Union(DifferentiableFunction,
                                            TwiceDifferentiableFunction),
                                   x::Vector,
                                   p::Vector,
                                   new_x::Vector,
                                   new_gradient::Vector;
                                   c1::Real = 1e-4,
                                   c2::Real = 0.9,
                                   rho::Real = 0.9,
                                   iterations::Integer = 1_000)

    # Keep track of the number of iterations
    iteration = 0

    # Keep track of calls to f and g
    f_calls, g_calls = 0, 0

    # Calculate parameter space size
    n = length(x)

    # Evaluate the function and gradient at current position
    f_calls += 1
    g_calls += 1
    f_x = d.fg!(x, new_gradient)
    gxp = dot(new_gradient, p)

    # The default step-size is always 1.0
    alpha = 1.0

    # Propose a new x after moving length alpha in direction p
    for i in 1:n
        new_x[i] = x[i] + alpha * p[i]
    end

    # Expand step-size
    # while (dot(new_gradient, p) < c2 * gxp) && (alpha < 65536.0)
    #     alpha *= 2.0
    #     for i in 1:n
    #         new_x[i] = x[i] + alpha * p[i]
    #     end
    #     g_calls += 1
    #     d.g!(new_x, new_gradient)
    # end

    # Keep coming closer to x until we find a point that is
    # as good as the gradient suggests we can achieve
    f_calls += 1
    f_new_x = d.f(new_x)
    while f_new_x > f_x + c1 * alpha * gxp
        iteration += 1
        if iteration > iterations
            error("Too many iterations in backtracking_line_search!")
        end
        alpha *= rho
        for i in 1:n
            new_x[i] = x[i] + alpha * p[i]
        end
        f_calls += 1
        f_new_x = d.f(new_x)
    end

    return alpha, f_calls, g_calls
end
