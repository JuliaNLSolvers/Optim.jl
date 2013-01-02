function two_loop(g_x::Vector,
                  rho::Vector,
                  s::Vector,
                  y::Vector,
                  m::Int64,
                  k::Int64)
    q = g_x

    index = min(k - 1, m)
    alpha = zeros(m)
    for i = index:-1:1
        alpha[i] = rho[i] * dot(s[i], q)
        q -= alpha[i] * y[i]
      end

    r = index == 0 ? q : q * dot(s[index], s[index]) / dot(s[index], s[index])

    for i = 1:index
        beta = rho[i] * dot(y[i], r)
        r += s[i] * (alpha[i] - beta)
    end

    -r
end

function l_bfgs(f::Function,
                g::Function,
                initial_x::Vector,
                m::Int64,
                tolerance::Float64,
                max_iterations::Int64,
                store_trace::Bool,
                show_trace::Bool)
    # Set iteration counter.
    k = 1

    # Set starting point. We will store x and new_x.
    x = initial_x
    new_x = initial_x

    # Establish size of parameter space.
    n = length(x)

    # Initialize rho, s and y.
    # Use Any arrays for the time being
    # Eventually move over to a faster Float64 representation
    # for which we build shift and push operations.
    rho = zeros(0)
    s = {}
    y = {}

    # Compute the initial gradient.
    g_x = g(x)

    # Print trace information
    tr = OptimizationTrace()

    # Iterate until convergence.
    converged = false

    while !converged && k <= max_iterations
        # Select a search direction.
        p = two_loop(g_x, rho, s, y, m, k)

        # Select a step-size.
        alpha = backtracking_line_search(f, g, x, p)

        # Show trace.
        if store_trace || show_trace
            d = Dict()
            d["g(x)"] = g_x
            d["Step-size"] = alpha
            d["First-order opt"] = norm(g_x, Inf)
            os = OptimizationState(x, f(x), k, d)
            if store_trace
                push(tr, os)
            end
            if show_trace
                println(os)
            end
        end

        # Update position.
        x_new = x + alpha * p

        # Estimate movement.
        tmp_s = x_new - x
        tmp_y = g(x_new) - g(x)
        tmp_rho = 1 / dot(tmp_y, tmp_s)
        if tmp_rho == Inf
            println("Cannot decrease the objective function along the current search direction")
            break
        end

        # Discard unneeded s, y and rho.
        if k > m
            shift(s)
            shift(y)
            shift(rho)
        end

        # Keep a record of the new s, y and rho.
        push(s, tmp_s)
        push(y, tmp_y)
        push(rho, tmp_rho)

        # Update our position.
        x = x_new
        g_x = g(x)

        # Update the iteration counter.
        k += 1

        # Assess convergence.
        if norm(g_x, Inf) <= tolerance
           converged = true
        end
    end

    OptimizationResults("L-BFGS", initial_x, x, f(x), k, converged, tr)
end

function l_bfgs(f::Function, g::Function, initial_x::Vector,
                store_trace::Bool, show_trace::Bool)
    l_bfgs(f, g, initial_x, 10, 10e-8, 1_000, store_trace, show_trace)
end
function l_bfgs(f::Function, g::Function, initial_x::Vector)
    l_bfgs(f, g, initial_x, 10, 10e-8, 1_000, false, false)
end
