function modindex(i::Integer, m::Integer)
    x = mod(i, m)
    if x == 0
        return m
    else
        return x
    end
end

function two_loop!(g_x::Vector,
                   rho::Vector,
                   s::Matrix,
                   y::Matrix,
                   m::Integer,
                   k::Integer,
                   alpha::Vector,
                   q::Vector,
                   p::Vector)
    # Copy gradient into q
    copy!(q, g_x)

    upper = k - 1
    lower = k - m
    for index = upper:-1:lower
        if index < 1
            continue
        end
        i = modindex(index, m)
        alpha[i] = rho[i] * dot(s[:, i], q)
        q -= alpha[i] * y[:, i]
    end

    # Copy q into p
    copy!(p, q)

    for index = lower:1:upper
        if index < 1
            continue
        end
        i = modindex(index, m)
        beta = rho[i] * dot(y[:, i], p)
        p[:] += s[:, i] * (alpha[i] - beta)
    end

    p[:] = -p

    return
end

function l_bfgs(d::DifferentiableFunction,
                initial_x::Vector,
                m::Integer,
                tolerance::Float64,
                max_iterations::Integer,
                store_trace::Bool,
                show_trace::Bool)
    # Set iteration counter
    k = 1

    # Set starting point
    # We will store x and new_x
    x = copy(initial_x)

    # Establish size of parameter space
    n = length(x)

    # Initialize rho, s and y
    rho = Array(Float64, m)
    s = Array(Float64, n, m)
    y = Array(Float64, n, m)
    q = Array(Float64, n)
    p = Array(Float64, n)
    tmp_s = Array(Float64, n)
    tmp_y = Array(Float64, n)

    # Re-use this vector during every call to two_loop()
    twoloop_v = Array(Float64, m)

    # Compute the initial values of f and g
    f_x = d.f(x)
    g_x = Array(Float64, n)
    g_new = Array(Float64, n)
    d.g!(x, g_new)
    copy!(g_x, g_new)

    # Print trace information
    tr = OptimizationTrace()

    # Iterate until convergence.
    converged = false

    while !converged && k <= max_iterations
        # Select a search direction
        two_loop!(g_new, rho, s, y, m, k, twoloop_v, q, p)

        # Select a step-size
        alpha, f_update, g_update = backtracking_line_search(d.f, d.g!, x, p)

        # Show trace
        if store_trace || show_trace
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

        # Update position
        tmp_s = alpha * p
        x = x + tmp_s
        copy!(g_x, g_new)
        d.g!(x, g_new)

        # Estimate movement
        for index in 1:n
            tmp_y[index] = g_new[index] - g_x[index]
        end
        tmp_rho = 1.0 / dot(tmp_y, tmp_s)
        if isinf(tmp_rho)
            println("Cannot decrease the objective function along the current search direction")
            break
        end

        # Keep a record of the new s, y and rho.
        s[:, modindex(k, m)] = tmp_s
        y[:, modindex(k, m)] = tmp_y
        rho[modindex(k, m)] = tmp_rho

        # Update the iteration counter
        k += 1

        # Assess convergence
        if norm(g_x, Inf) <= tolerance
           converged = true
        end
    end

    OptimizationResults("L-BFGS", initial_x, x, f_x, k, converged, tr)
end

function l_bfgs(f::Function, g!::Function, initial_x::Vector,
                store_trace::Bool, show_trace::Bool)
    l_bfgs(f, g!, initial_x, 10, 10e-8, 1_000, store_trace, show_trace)
end
function l_bfgs(f::Function, g!::Function, initial_x::Vector)
    l_bfgs(f, g!, initial_x, 10, 10e-8, 1_000, false, false)
end
