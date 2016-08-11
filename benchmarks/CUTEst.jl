##########################################################################
#
# Benchmark optimization algorithms by tracking:
#
# * Number of iterations
# * Number of f_calls
# * Number of g_calls
# * Euclidean error of solution
# * Memory requirements (TODO)
#
##########################################################################
using CUTEst

min_dim = 1
max_dim = 100
n = length(default_solvers)
m = length(cutest_problems)
f = open(join([version_dir, "cutest_benchmark.csv"], "/"), "w")
write(f, join(["Problem", "Optimizer", "Converged", "Time", "Minimum", "Iterations", "f_calls", "g_calls", "f_hat", "f_error", "x_error"], ","))
write(f, "\n")
@showprogress 1 "Benchmarking..." for p in cutest_problems
    output = []
    nlp = CUTEstModel(p)
    x_hat = copy(nlp.meta.x0)
    f_hat = obj(nlp, x_hat)
    xs = []
    times = []
    if nlp.meta.nvar <= max_dim && min_dim <= nlp.meta.nvar
        for i = 1:n
            if !(default_solvers[i] in (Newton(), NewtonTrustRegion()))
            	try
            		result = optimize(x->obj(nlp, x),(x, stor) -> grad!(nlp,x,stor), nlp.meta.x0, default_solvers[i], OptimizationOptions(g_tol=1e-16))
                    mintime = @elapsed optimize(x->obj(nlp, x),(x, stor) -> grad!(nlp,x,stor), nlp.meta.x0, default_solvers[i], OptimizationOptions(g_tol=1e-16))
                    if f_hat > Optim.minimum(result)
                        f_hat = Optim.minimum(result)
                        x_hat[:] = Optim.minimizer(result)
                    end
                    push!(output, [p,
                                   default_names[i],
                                   Optim.converged(result),
                                   mintime,
                                   Optim.minimum(result),
                                   Optim.iterations(result),
                                   Optim.f_calls(result),
                                   Optim.g_calls(result)])
                    push!(xs, Optim.minimizer(result))
                catch
                    push!(output, ([p,
                                   default_names[i],
                                   false,
                                   Inf,
                                   Inf,
                                   Inf,
                                   Inf,
                                   Inf]))
                    push!(xs, fill(Inf, length(nlp.meta.x0)))
            	end
            end
        end
        for i = 1:n
            if !(default_solvers[i] in (Newton(), NewtonTrustRegion())) # Assuming second order methods are at the bottom
                write(f, join(map(x->"$x",output[i]),",")*",")
                write(f, join(map(x->"$x",[f_hat, output[i][5]-f_hat, norm(xs[i]-x_hat, Inf)]),","))
                write(f, "\n")
            end
        end
    end
    cutest_finalize(nlp)
end

close(f)
