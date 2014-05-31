function lmfit(f::Function, p0; kwargs...)
	# minimizes f(p) using leastsq()
	# returns p, f(p), g(p) where
	#   p - best fit parameters
	#   f(p) - vector of function at p, typically residuals
	#   g(p) - estimated Jacobian at p

	# this is a convenience function for the below curve_fit() methods
	# which assume f(p) is the cost function, i.e. the residual of a 
	# model where model(xpts, params...) = ydata + noise

	# construct Jacobian function
	g = Calculus.jacobian(f)

	results = levenberg_marquardt(f, g, p0; kwargs...)
	p = results.minimum
	return p, f(p), g(p)
end

function curve_fit(model::Function, xpts, ydata, p0; kwargs...)
	# minimizes model(p) = sum(ydata - model(xpts)).^2 using leastsq()

	# construct the cost function
	f(p) = model(xpts, p) - ydata
	lmfit(f,p0; kwargs...)
end

function curve_fit( model::Function, xpts, ydata, wt::Vector, p0; kwargs...)
	# use a per bin weight, e.g. wt = 1/sigma
	f(p) = wt .* ( model(xpts, p) - ydata )
	lmfit(f,p0; kwargs...)
end

function curve_fit( model::Function, xpts, ydata, wt::Matrix, p0; kwargs... )
	# use a matrix weight, e.g. inverse_covariance 

	# Cholesky is effectively a sqrt of a matrix,  which is what we want 
	# to minimize in the least-squares of levenberg_marquardt()
	u = chol(wt)  # requires matrix to be positive definite

	f(p) = u * ( model(xpts, p) - ydata )
	lmfit(f,p0; kwargs...)
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
