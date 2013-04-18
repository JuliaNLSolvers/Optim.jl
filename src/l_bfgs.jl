function modindex(i::Integer, m::Integer)
    x = mod(i, m)
    if x == 0
        return m
    else
        return x
    end
end

function twoloop!(g_x::Vector,
                  rho::Vector,
                  s::Matrix,
                  y::Matrix,
                  m::Integer,
                  k::Integer,
                  alpha::Vector,
                  q::Vector,
                  p::Vector)
    # Copy gradient into q for backward pass
    copy!(q, g_x)

    n = length(p)
    @assert length(q) == n

    upper = k - 1
    lower = k - m
    for index in upper:-1:lower
        if index < 1
            continue
        end
        i = modindex(index, m)
        alpha[i] = rho[i] * dot(s[:, i], q)
        for j in 1:n
            q[j] -= alpha[i] * y[j, i]
        end
    end

    # Copy q into p for forward pass
    copy!(p, q)

    for index in lower:1:upper
        if index < 1
            continue
        end
        i = modindex(index, m)
        beta = rho[i] * dot(y[:, i], p)
        for j in 1:n
            p[j] += s[j, i] * (alpha[i] - beta)
        end
    end

    for j in 1:n
        p[j] = -1.0 * p[j]
    end

    return
end

function l_bfgs_trace!(tr::OptimizationTrace,
                       x::Vector,
                       f_x::Real,
                       g_new::Vector,
                       k::Integer,
                       alpha::Real,
                       store_trace::Bool,
                       show_trace::Bool)
    dt = Dict()
    dt["g(x)"] = copy(g_new)
    dt["Step-size"] = alpha
    dt["First-order opt"] = norm(g_new, Inf)
    os = OptimizationState(x, f_x, k, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function l_bfgs(d::DifferentiableFunction,
                initial_x::Vector;
                m::Integer = 10,
                tolerance::Real = 1e-8,
                iterations::Integer = 1_000,
                store_trace::Bool = false,
                show_trace::Bool = false)

    # Set iteration counter
    k = 0

    # Keep a record of the starting point
    x = copy(initial_x)

    # Establish size of parameter space
    n = length(x)

    # Initialize rho, s and y
    rho = Array(Float64, m)
    s = Array(Float64, n, m)
    y = Array(Float64, n, m)

    # Initialize q, p, tmp_s and tmp_y
    q = Array(Float64, n)
    p = Array(Float64, n)
    tmp_s = Array(Float64, n)
    tmp_y = Array(Float64, n)

    # Reuse storage during calls to backtracking line search
    ls_x = Array(Float64, n)
    ls_gradient = Array(Float64, n)

    # Re-use this vector during every call to two_loop()
    twoloop_v = Array(Float64, m)

    # Compute the initial values of f and g
    g_x = Array(Float64, n)
    g_new = Array(Float64, n)
    f_calls = 1
    g_calls = 1
    f_x = d.fg!(x, g_new)
    copy!(g_x, g_new)

    # Print trace information
    tr = OptimizationTrace()

    # Show trace
    if store_trace || show_trace
        l_bfgs_trace!(tr, x, f_x, g_new, k, 0.0, store_trace, show_trace)
    end

    # Iterate until convergence
    converged = false

    while !converged && k <= iterations
        # Update count
        k += 1

        # Select a search direction
        twoloop!(g_new, rho, s, y, m, k, twoloop_v, q, p)

        # Select a step-size
        alpha, f_update, g_update =
          backtracking_line_search!(d, x, p, ls_x, ls_gradient)
        f_calls += f_update
        g_calls += g_update

        # Update position
        for i in 1:n
            tmp_s[i] = alpha * p[i]
            x[i] = x[i] + tmp_s[i]
        end
        copy!(g_x, g_new)
        f_x = d.fg!(x, g_new)
        f_calls += 1
        g_calls += 1

        # Estimate movement
        for i in 1:n
            tmp_y[i] = g_new[i] - g_x[i]
        end
        tmp_rho = 1.0 / dot(tmp_y, tmp_s)
        if isinf(tmp_rho)
            @printf "Cannot decrease the objective function along the current search direction\n"
            break
        end

        # Keep a record of the new s, y and rho
        s[:, modindex(k, m)] = tmp_s
        y[:, modindex(k, m)] = tmp_y
        rho[modindex(k, m)] = tmp_rho

        # Assess convergence
        if norm(g_x, Inf) <= tolerance
           converged = true
        end

        # Show trace
        if store_trace || show_trace
            l_bfgs_trace!(tr, x, f_x, g_new, k, store_trace, show_trace)
        end
    end

    OptimizationResults("L-BFGS",
                        initial_x,
                        x,
                        f_x,
                        k,
                        converged,
                        tr,
                        f_calls,
                        g_calls)
end
