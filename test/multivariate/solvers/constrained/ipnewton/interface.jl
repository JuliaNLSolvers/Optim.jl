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