immutable LevenbergMarquardt <: Optimizer end

function levenberg_marquardt{T}(f::Function, g::Function, x0::AbstractVector{T};
    tolX::Real = 1e-8, tolG::Real = 1e-12, maxIter::Integer = 100,
    lambda::Real = 10.0, show_trace::Bool = false)
    # finds argmin sum(f(x).^2) using the Levenberg-Marquardt algorithm
    #          x
    # The function f should take an input vector of length n and return an output vector of length m
    # The function g is the Jacobian of f, and should be an m x n matrix
    # x0 is an initial guess for the solution
    # fargs is a tuple of additional arguments to pass to f
    # available options:
    #   tolX - search tolerance in x
    #   tolG - search tolerance in gradient
    #   maxIter - maximum number of iterations
    #   lambda - (inverse of) initial trust region radius
    #   show_trace - print a status summary on each iteration if true
    # returns: x, J
    #   x - least squares solution for x
    #   J - estimate of the Jacobian of f at x

    # other constants
    const MAX_LAMBDA = 1e16 # minimum trust region radius
    const MIN_LAMBDA = 1e-16 # maximum trust region radius
    const MIN_STEP_QUALITY = 1e-3
    const GOOD_STEP_QUALITY = 0.75
    const MIN_DIAGONAL = 1e-6 # lower bound on values of diagonal matrix used to regularize the trust region step


    converged = false
    x_converged = false
    g_converged = false
    need_jacobian = true
    iterCt = 0
    x = copy(x0)
    delta_x = copy(x0)
    f_calls = 0
    g_calls = 0

    fcur = f(x)
    f_calls += 1
    residual = sumabs2(fcur)

    # Create buffers
    m = length(fcur)
    n = length(x)
    JJ = Matrix{T}(n, n)
    n_buffer = Vector{T}(n)
    m_buffer = Vector{T}(m)

    # Maintain a trace of the system.
    tr = OptimizationTrace(LevenbergMarquardt())
    if show_trace
        d = Dict("lambda" => lambda)
        os = OptimizationState(iterCt, sumabs2(fcur), NaN, d)
        push!(tr, os)
        println(os)
    end

    while (~converged && iterCt < maxIter)
        if need_jacobian
            J = g(x)
            g_calls += 1
            need_jacobian = false
        end
        # we want to solve:
        #    argmin 0.5*||J(x)*delta_x + f(x)||^2 + lambda*||diagm(J'*J)*delta_x||^2
        # Solving for the minimum gives:
        #    (J'*J + lambda*diagm(DtD)) * delta_x == -J^T * f(x), where DtD = sumabs2(J,1)
        # Where we have used the equivalence: diagm(J'*J) = diagm(sumabs2(J,1))
        # It is additionally useful to bound the elements of DtD below to help
        # prevent "parameter evaporation".
        DtD = vec(sumabs2(J, 1))
        for i in 1:length(DtD)
            if DtD[i] <= MIN_DIAGONAL
                DtD[i] = MIN_DIAGONAL
            end
        end
        # delta_x = ( J'*J + lambda * diagm(DtD) ) \ ( -J'*fcur )
        At_mul_B!(JJ, J, J)
        @simd for i in 1:n
            @inbounds JJ[i, i] += lambda * DtD[i]
        end
        At_mul_B!(n_buffer, J, fcur)
        scale!(n_buffer, -1)
        delta_x = JJ \ n_buffer

        # if the linear assumption is valid, our new residual should be:
        # predicted_residual = sumabs2(J*delta_x + fcur)
        A_mul_B!(m_buffer, J, delta_x)
        LinAlg.axpy!(1, fcur, m_buffer)
        predicted_residual = sumabs2(m_buffer)
        # check for numerical problems in solving for delta_x by ensuring that the predicted residual is smaller
        # than the current residual
        if predicted_residual > residual + 2max(eps(predicted_residual),eps(residual))
            warn("""Problem solving for delta_x: predicted residual increase.
                             $predicted_residual (predicted_residual) >
                             $residual (residual) + $(eps(predicted_residual)) (eps)""")
        end
        # try the step and compute its quality
        @simd for i in 1:n
            @inbounds n_buffer[i] = x[i] + delta_x[i]
        end
        trial_f = f(n_buffer)
        f_calls += 1
        trial_residual = sumabs2(trial_f)
        # step quality = residual change / predicted residual change
        rho = (trial_residual - residual) / (predicted_residual - residual)
        if rho > MIN_STEP_QUALITY
            x += delta_x
            fcur = trial_f
            residual = trial_residual
            if rho > GOOD_STEP_QUALITY
                # increase trust region radius
                lambda = max(0.1*lambda, MIN_LAMBDA)
            end
            need_jacobian = true
        else
            # decrease trust region radius
            lambda = min(10*lambda, MAX_LAMBDA)
        end
        iterCt += 1

        # show state
        if show_trace
            At_mul_B!(n_buffer, J, fcur)
            g_norm = norm(n_buffer, Inf)
            d = @compat Dict("g(x)" => g_norm, "dx" => delta_x, "lambda" => lambda)
            os = OptimizationState(iterCt, sumabs2(fcur), g_norm, d)
            push!(tr, os)
            println(os)
        end

        # check convergence criteria:
        # 1. Small gradient: norm(J^T * fcur, Inf) < tolG
        # 2. Small step size: norm(delta_x) < tolX
        At_mul_B!(n_buffer, J, fcur)
        if norm(n_buffer, Inf) < tolG
            g_converged = true
        elseif norm(delta_x) < tolX*(tolX + norm(x))
            x_converged = true
        end
        converged = g_converged | x_converged
    end

    MultivariateOptimizationResults("Levenberg-Marquardt", x0, x, sumabs2(fcur), iterCt, !converged, x_converged, 0.0, false, 0.0, g_converged, tolG, tr, f_calls, g_calls)
end
