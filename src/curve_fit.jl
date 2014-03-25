function curve_fit(model::Function, xpts, ydata, p0; kwargs...)
	# assumes model(xpts, params...) = ydata + noise
	# minimizes F(p) = sum(ydata - f(xdata)).^2 using leastsq()
	# returns p, f(p), g(p) where
	#   p - best fit parameters
	#   f(p) - vector of residuals
	#   g(p) - estimated Jacobian at p

	# construct the cost function
	f(p) = model(xpts, p) - ydata
	
	# construct Jacobian function
	g = Calculus.jacobian(f)

	results = levenberg_marquardt(f, g, p0; kwargs...)
	p = results.minimum
	return p, f(p), g(p)
end

estimate_errors(p, residuals, J) = estimate_errors(p, residuals, J, .95)

function estimate_errors(p, residuals, J, alpha)
	# estimate_errors(p, residuals, J, alpha) computes (1-alpha) error estimates for the parameters from leastsq
	#   p - parameters
	#   residuals - vector of residuals
	#   J - Jacobian
	#   alpha - compute alpha percent confidence interval, (e.g. alpha=0.95 for 95% CI)

	# mean square error is: standard square error / degrees of freedom
	n, p = size(J)
	mse = sse(residuals)/(n-p)

	# compute the covariance matrix from the QR decomposition
	Q,R = qr(J)
	Rinv = inv(R)
	covar = Rinv*Rinv'*mse

	# then the standard errors are given by the sqrt of the diagonal
	std_error = sqrt(diag(covar))

	# scale by quantile of the student-t distribution
	dist = TDist(n-p)
	std_error *= quantile(dist, alpha)
end