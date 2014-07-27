# Because MinFinder is heuristic, allow several times to test for each problem

runs = 10

for (name, problem) in Optim.MultipleMinimaProblems.examples

	Nmin = Int[] # number of minima found

	for unused = 1:runs
		runmin = 0
		mins, fcount, search, iters = minfinder(problem.f, problem.l, problem.u)
		for i = 1:length(mins)-1
			#check to see if minimum is an expected minimum
			any([norm(mins[i].x - problem.minima[j],2) < 1e3 for j in 
				problem.minima]) &&
			#check if minimum not found more than once
			!any([norm(mins[i].x - mins[j].x, 2) < 1e3 for j = 
				i+1:length(mins)]) && runmin += 1
		end
		# last minimum only to check against expected minima
		any([norm(mins[end].x - problem.minima[j],2) <1e3 for j in 
				problem.minima]) && runmin += 1

		push!(Nmin, runmin)
	end

	@assert maximum(Nmin) length(problem.minima)
end
println("success!")