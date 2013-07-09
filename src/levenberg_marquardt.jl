# sse(x) gives the L2 norm of x
sse(x) = (x'*x)[1]

function levenberg_marquardt(f::Function, g::Function, x0; tolX=1e-8, tolG=1e-12, maxIter=100, lambda=100.0, show_trace=false)
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
	const MIN_DIAGONAL = 1e-6 # lower bound on values of diagonal matrix used to regularize the trust region step

	converged = false
	iterCt = 0
	x = x0
	delta_x = copy(x0)
	f_calls = 0
	g_calls = 0

	fcur = f(x)
	f_calls += 1
	residual = sse(fcur)
	
	# Maintain a trace of the system.
	tr = OptimizationTrace()
	if show_trace
		d = {"lambda" => lambda}
		os = OptimizationState(x, sse(fcur), iterCt, d)
		push!(tr, os)
		println(os)
	end

	while ( ~converged && iterCt < maxIter )
		J = g(x)
		g_calls += 1
		# we want to solve:
		#    argmin 0.5*||J(x)*delta_x + f(x)||^2 + lambda*||diagm(J'*J)*delta_x||^2
		# Solving for the minimum gives:
		#    (J'*J + lambda*DtD) * delta_x == -J^T * f(x), where DtD = diagm(sum(J.^2,1))
		# Where we have used the equivalence: diagm(J'*J) = diagm(sum(J.^2, 1))
		# It is additionally useful to bound the elements of DtD below to help
		# prevent "parameter evaporation".
		DtD = diagm(Float64[max(x, MIN_DIAGONAL) for x in sum(J.^2,1)])
		delta_x = ( J'*J + sqrt(lambda)*DtD ) \ -J'*fcur
		# if the linear assumption is valid, our new residual should be:
		predicted_residual = sse(J*delta_x + fcur)
		# check for numerical problems in solving for delta_x by ensuring that the predicted residual is smaller
		# than the current residual
		if predicted_residual > residual
			error("Error solving for delta_x: predicted residual increase.")
		end
		# try the step and compute its quality
		trial_f = f(x + delta_x)
		f_calls += 1
		trial_residual = sse(trial_f)
		# step quality = residual change / predicted residual change
		rho = (trial_residual - residual) / (predicted_residual - residual)

		if rho > MIN_STEP_QUALITY
			x += delta_x
			fcur = trial_f
			residual = trial_residual
			# increase trust region radius
			lambda = max(0.1*lambda, MIN_LAMBDA)
		else
			# decrease trust region radius
			lambda = min(10*lambda, MAX_LAMBDA)
		end
		iterCt += 1

		# show state
		if show_trace
			d = {"g(x)" => norm(J'*fcur, Inf), "dx" => delta_x, "lambda" => lambda}
			os = OptimizationState(x, sse(fcur), iterCt, d)
			push!(tr, os)
			println(os)
		end

		# check convergence criteria:
		# 1. Small gradient: norm(J^T * fcur, Inf) < tolG
		# 2. Small step size: norm(delta_x) < tolX
		if norm(J' * fcur, Inf) < tolG
			converged = true
		elseif norm(delta_x) < tolX*(tolX + norm(x))
			converged = true
		end
	end

	# give the user info about the stopping condition
	if ~converged
		println("Exceeded maximum number of iterations")
	end

	MultivariateOptimizationResults("Levenberg-Marquardt", x0, x, sse(fcur), iterCt, !converged, false, 0.0, false, 0.0, converged, tolG, tr, f_calls, g_calls)
end