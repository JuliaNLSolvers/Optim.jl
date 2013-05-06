# Notational note
# JMW's dx_history <=> NW's S
# JMW's dgr_history <=> NW's Y

function modindex(i::Integer, m::Integer)
    x = mod(i, m)
    if x == 0
        return m
    else
        return x
    end
end

# Here alpha is a cache that parallels betas
# It is not the step-size
# q is also a cache
function twoloop!(s::Vector,
                  gr::Vector,
                  rho::Vector,
                  dx_history::Matrix,
                  dgr_history::Matrix,
                  m::Integer,
                  iteration::Integer,
                  alpha::Vector,
                  q::Vector)
    # Count number of parameters
    n = length(s)

    # Determine lower and upper bounds for loops
    lower = iteration - m
    upper = iteration - 1

    # Copy gr into q for backward pass
    copy!(q, gr)

    # Backward pass
    for index in upper:-1:lower
        if index < 1
            continue
        end
        i = modindex(index, m)
        alpha[i] = rho[i] * dot(dx_history[:, i], q)
        for j in 1:n
            q[j] -= alpha[i] * dgr_history[j, i]
        end
    end

    # Copy q into s for forward pass
    copy!(s, q)

    # Forward pass
    for index in lower:1:upper
        if index < 1
            continue
        end
        i = modindex(index, m)
        beta = rho[i] * dot(dgr_history[:, i], s)
        for j in 1:n
            s[j] += dx_history[j, i] * (alpha[i] - beta)
        end
    end

    # Negate search direction
    for i in 1:n
        s[i] = -1.0 * s[i]
    end

    return
end

function l_bfgs_trace!(tr::OptimizationTrace,
                       x::Vector,
                       f_x::Real,
                       gr::Vector,
                       alpha::Real,
                       iteration::Integer,
                       store_trace::Bool,
                       show_trace::Bool)
    dt = Dict()
    dt["g(x)"] = copy(gr)
    dt["Current step size"] = alpha
    dt["Maximum component of g(x)"] = norm(gr, Inf)
    os = OptimizationState(copy(x), f_x, iteration, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        println(os)
    end
end

function l_bfgs{T}(d::Union(DifferentiableFunction,
                            TwiceDifferentiableFunction),
                   initial_x::Vector{T};
                   m::Integer = 10,
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

    # Maintain current gradient in gr and previous gradient in gr_previous
    gr = Array(Float64, n)
    gr_previous = Array(Float64, n)

    # Store a history of changes in position and gradient
    rho = Array(Float64, m)
    dx_history = Array(Float64, n, m)
    dgr_history = Array(Float64, n, m)

    # The current search direction
    s = Array(Float64, n)

    # Buffers for use in line search
    x_ls = Array(Float64, n)
    gr_ls = Array(Float64, n)

    # Store f(x) in f_x
    f_x = d.fg!(x, gr)
    f_calls += 1
    g_calls += 1
    copy!(gr_previous, gr)

    # Store the history of function values
    f_values = Array(T, iterations + 1)
    fill!(f_values, nan(T))
    f_values[iteration + 1] = f_x

    # Keep track of step-sizes
    alpha = alphainit(1.0, x, gr, f_x)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Buffers for new entries of dx_history and dgr_history
    dx = Array(Float64, n)
    dgr = Array(Float64, n)

    # Buffers for use by twoloop!
    twoloop_q = Array(Float64, n)
    twoloop_alpha = Array(Float64, m)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace
    if tracing
        l_bfgs_trace!(tr, x, f_x, gr, alpha,
                      iteration, store_trace, show_trace)
    end

    # Iterate until convergence
    f_converged = false
    gr_converged = false
    converged = false
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # Determine the L-BFGS search direction
        twoloop!(s, gr, rho, dx_history, dgr_history, m, iteration,
                 twoloop_alpha, twoloop_q)

        # Refresh the line search cache
        dphi0 = dot(gr, s)
        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          linesearch!(d, x, s, x_ls, gr_ls, lsr, alpha, mayterminate)
        f_calls += f_update
        g_calls += g_update

        # Update current position
        for i in 1:n
            dx[i] = alpha * s[i]
            x[i] = x[i] + dx[i]
        end

        # Maintain a record of the previous gradient
        copy!(gr_previous, gr)

        # Update the function value and gradient
        f_x = d.fg!(x, gr)
        f_calls += 1
        g_calls += 1
        f_values[iteration + 1] = f_x

        # Measure the change in the gradient
        for i in 1:n
            dgr[i] = gr[i] - gr_previous[i]
        end

        # Update the L-BFGS history of positions and gradients
        rho_iteration = 1.0 / dot(dx, dgr)
        if isinf(rho_iteration)
            # TODO: Introduce a formal error? There was a warning here previously
            break
        end
        dx_history[:, modindex(iteration, m)] = dx
        dgr_history[:, modindex(iteration, m)] = dgr
        rho[modindex(iteration, m)] = rho_iteration

        # Assess convergence
        if norm(gr, Inf) < tolerance
            gr_converged = true
        end
        if abs(f_values[iteration + 1] - f_values[iteration]) < 1e-32
            f_converged = true
        end
        converged = gr_converged || f_converged

        # Show trace
        if tracing
            l_bfgs_trace!(tr, x, f_x, gr, alpha,
                          iteration, store_trace, show_trace)
        end
    end

    OptimizationResults("L-BFGS",
                        initial_x,
                        x,
                        f_x,
                        iteration,
                        f_converged,
                        gr_converged,
                        tr,
                        f_calls,
                        g_calls,
                        f_values[1:(iteration + 1)])
end
