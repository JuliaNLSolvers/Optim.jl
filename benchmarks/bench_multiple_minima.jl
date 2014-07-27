#######################
# Benchmark MinFinder #
#######################

using Optim

# Print out a header line for the TSV-formatted report
println(join({"Problem", "avg_Nmin", "max_Nmin", "min_Nmin", "avg_fcount",
          "avg_searches", "avg_iterations"}, "\t"))

for (name, problem) in Optim.MultipleMinimaProblems.examples

	# Force compilation
    results = minfinder(problem.f, problem.l, problem.u)

    n = 50 # minfinder repititions, n=50 in paper

	Nmin = Int[]
	Nfcount = Int[]
	Nsearches = Int[]
	Niterations = Int[]

	for unused = 1:n
		mins,fcount,search,iters=minfinder(problem.f, problem.l, problem.u)
		push!(Nmin, length(mins))
		push!(Nfcount, fcount)
		push!(Nsearches, search)
		push!(Niterations, iters)
	end

	println(join({name, mean(Nmin), maximum(Nmin), minimum(Nmin), mean(Nfcount), 
		mean(Nsearches), mean(Niterations)}, "\t"))
end

