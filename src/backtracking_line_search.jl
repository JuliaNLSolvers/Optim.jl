function backtracking_line_search(f::Function,
                                  g!::Function,
                                  x::Vector,
                                  p::Vector,
                                  c1::Float64,
                                  c2::Float64,
                                  rho::Float64,
                                  max_iterations::Integer)

    # Keep track of the number of iterations
    i = 0

    # Keep track of calls to f and g
    f_calls, g_calls = 0, 0

    # Store a copy of the function and gradient evaluated at x
    n = length(x)
    new_x = Array(Float64, n)
    new_gradient = Array(Float64, n)
    f_calls += 1
    g_calls += 1
    f_x = f(x)
    g!(x, new_gradient)
    gxp = dot(new_gradient, p)

    # The default step-size is always 1
    alpha = 1.0

    # Propose a new x after moving length alpha in direction p
    for index in 1:n
        new_x[index] = x[index] + alpha * p[index]
    end

    # Expand step-size until f starts to increase
    # NB: This is very expensive if the gradient costs much more than f(x)
    # In my tests, it also increases the number of steps, not decreases it
    # while (dot(new_gradient, p) < c2 * gxp) && (alpha < 65536.0)
    #     alpha *= 2.0
    #     for index in 1:n
    #         new_x[index] = x[index] + alpha * p[index]
    #     end
    #     g_calls += 1
    #     g!(new_x, new_gradient)
    # end

    # Keep coming closer to x until we find a point that is as good
    # as the gradient suggests we can achieve
    f_calls += 1
    f_new_x = f(new_x)
    while f_new_x > f_x + c1 * alpha * gxp
        alpha *= rho
        for index in 1:n
            new_x[index] = x[index] + alpha * p[index]
        end
        i += 1
        if i > max_iterations
            error("Too many iterations in backtracking_line_search")
        end
        f_calls += 1
        f_new_x = f(new_x)
    end

    return alpha, f_calls, g_calls
end

function backtracking_line_search(f::Function,
                                  g!::Function,
                                  x::Vector,
                                  p::Vector)
    backtracking_line_search(f, g!, x, p, 1e-6, 0.9, 0.9, 1_000)
end
