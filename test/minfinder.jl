# Because MinFinder is heuristic, allow several times to test for each problem

runs = 10

for (name, problem) in Optim.MultipleMinimaProblems.examples

	Nmin = zeros(Int,1) # number of minima found

	for unused = 1:runs
		runmin = 0
		
		mins, fcount, search, iters = minfinder(problem.f, problem.l, problem.u)

		for i = 1:length(mins)-1
			#check to see if minimum is an expected minimum
			#println([norm(mins[i].x - m, 2) < 1e-3 for m in problem.minima])
			# println(any([norm(mins[i].x - mins[j].x, 2) < 1e3 for j = 
			# 			(i+1):length(mins)])
			if any([norm(mins[i].x - m, 2) < 1e-3 for m in problem.minima]) &&
			#check if minimum not found more than once
					!any([norm(mins[i].x - mins[j].x, 2) < 1e-3 for j = 
						(i+1):length(mins)])

				runmin += 1
			end
		end
		# last minimum only to check against expected minima
		if any([norm(mins[end].x - m, 2) < 1e-3 for m in problem.minima])
			runmin += 1
		end
		push!(Nmin, runmin)
		@printf "%s out of %s minima ok\n" runmin length(mins)
		 # push!(Nmin, length(mins))
	end

	# @printf " test if %d == %d ?" maximum(Nmin) length(problem.minima)
	@assert maximum(Nmin)==length(problem.minima)
end

#println("success!")