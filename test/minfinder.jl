
for (name, problem) in Optim.MultipleMinimaProblems.examples
    @printf "%s \n" name

    srand(1234)
    mins, fcount, search, iters = minfinder(problem.f, problem.l, problem.u)
    @assert length(mins)==length(problem.minima)

    for m in mins
        foundmin = false
        for i in 1:length(problem.minima)
            if norm(m.x - problem.minima[i], 2) < 1e-5; foundmin = true; end
        end
        foundmin || println(name, m)
        @assert foundmin
    end

end
