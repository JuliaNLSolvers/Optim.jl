@testset "#711" begin
	# make sure it doesn't try to promote df
	dof = 7

	fun(x) = 0.0; x0 = fill(0.1, dof)
	df = TwiceDifferentiable(fun, x0)

	lx = fill(-1.2, dof); ux = fill(+1.2, dof)
	dfc = TwiceDifferentiableConstraints(lx, ux)

	res = optimize(df, dfc, x0, IPNewton(); autodiff=:forward)
	res = optimize(df, dfc, x0, IPNewton())
end

@testset "#600" begin
	function exponential(x)
		return exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
	end

	function exponential_gradient!(storage, x)
		storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
		storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
		storage
	end

	function exponential_gradient(x)
		storage = similar(x)
		storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
		storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
		storage
	end

	initial_x = [0.0, 0.0]
	optimize(exponential, exponential_gradient!, initial_x, BFGS())
	lb = fill(-0.1, 2)
	ub = fill(1.1, 2)
	od = OnceDifferentiable(exponential, initial_x)
	optimize(od, lb, ub, initial_x, IPNewton())
end