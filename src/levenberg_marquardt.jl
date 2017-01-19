immutable LevenbergMarquardt <: Optimizer end

"""
    `levenberg_marquardt(f, g, initial_x; <keyword arguments>`

Returns the argmin over x of `sum(f(x).^2)` using the Levenberg-Marquardt algorithm, and an
estimate of the Jacobian of `f` at x.

The function `f` should take an input vector of length n and return an output vector of length m.
The function `g` is the Jacobian of f, and should be an m x n matrix.
`initial_x` is an initial guess for the solution.

Implements box constraints as described in Kanzow, Yamashita, Fukushima (2004; J Comp & Applied Math).

# Keyword arguments
* `tolX::Real=1e-8`: search tolerance in x
* `tolG::Real=1e-12`: search tolerance in gradient
* `maxIter::Integer=100`: maximum number of iterations
* `lambda::Real=10.0`: (inverse of) initial trust region radius
* `show_trace::Bool=false`: print a status summary on each iteration if true
* `lower,upper=[]`: bound solution to these limits
"""
function levenberg_marquardt{F<:Function, G<:Function, T}(f::F, g::G, initial_x::AbstractVector{T};
    tolX::Real = 1e-8, tolG::Real = 1e-12, maxIter::Integer = 100,
    lambda::Real = 10.0, show_trace::Bool = false, lower::Vector{T} = Array{T}(0), upper::Vector{T} = Array{T}(0))

    if !has_deprecated_levenberg_marquardt[]
        warn("levenberg_marquardt has been moved out of Optim.jl and into LsqFit.jl. Please adjust your code, and change your dependency to match this migration.")
        has_deprecated_levenberg_marquardt[] = true
    end
    # check parameters
    ((isempty(lower) || length(lower)==length(initial_x)) && (isempty(upper) || length(upper)==length(initial_x))) ||
            throw(ArgumentError("Bounds must either be empty or of the same length as the number of parameters."))
    ((isempty(lower) || all(initial_x .>= lower)) && (isempty(upper) || all(initial_x .<= upper))) ||
            throw(ArgumentError("Initial guess must be within bounds."))

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
    x = copy(initial_x)
    delta_x = copy(initial_x)
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
    tr = OptimizationTrace{LevenbergMarquardt}()
    if show_trace
        d = Dict("lambda" => lambda)
        os = OptimizationState{LevenbergMarquardt}(iterCt, sumabs2(fcur), NaN, d)
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

        # apply box constraints
        if !isempty(lower)
            @simd for i in 1:n
               @inbounds delta_x[i] = max(x[i] + delta_x[i], lower[i]) - x[i]
            end
        end
        if !isempty(upper)
            @simd for i in 1:n
               @inbounds delta_x[i] = min(x[i] + delta_x[i], upper[i]) - x[i]
            end
        end

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
            os = OptimizationState{LevenbergMarquardt}(iterCt, sumabs2(fcur), g_norm, d)
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

    MultivariateOptimizationResults("Levenberg-Marquardt", initial_x, x, sumabs2(fcur), iterCt, !converged, x_converged, 0.0, false, 0.0, g_converged, tolG, false, tr, f_calls, g_calls, 0)
end
