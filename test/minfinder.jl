
for (name, problem) in Optim.MultipleMinimaProblems.examples

	srand(1234)
	mins, fcount, search, iters = minfinder(problem.f, problem.l, problem.u)
	@assert length(mins)==length(problem.minima)

end
