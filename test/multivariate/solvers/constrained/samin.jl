@testset "SAMIN" begin
    using Optim, Distributed
    prob = MVP.UnconstrainedProblems.examples["Himmelblau"]

    xtrue = prob.solutions
    f = OptimTestProblems.MultivariateProblems.objective(prob)
    x0 = prob.initial_x
    res = optimize(f, x0.-100., x0.+100.0, x0, Optim.SAMIN(), Optim.Options(iterations=1000000))
    @test Optim.minimum(res) < 1e-6

    #test distributed evaluation
    w= addprocs(4)
    @everywhere begin
        
        using Optim, OptimTestProblems, OptimTestProblems.MultivariateProblems
        const MVP = MultivariateProblems
        prob = MVP.UnconstrainedProblems.examples["Himmelblau"]
        xtrue = prob.solutions
        f = OptimTestProblems.MultivariateProblems.objective(prob)
        x0 = prob.initial_x
    end 
    res = optimize(f, x0.-100., x0.+100.0, x0, Optim.SAMIN(workers=w), Optim.Options(iterations=1000000))
    @test Optim.minimum(res) < 1e-6
    rmprocs(w)
end
#=
using BenchmarkTools, Plots, Distributed, Statistics

@everywhere using OptimTestProblems, OptimTestProblems.MultivariateProblems

1

end
samples=50
wrkrs=8
iters=1000

addprocs(wrkrs)
w=workers()
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere include("src/Optim.jl")
const MVP = MultivariateProblems
@everywhere begin
    const MVP = MultivariateProblems
    prob = MVP.UnconstrainedProblems.examples["Himmelblau"]

    xtrue = prob.solutions
    f = OptimTestProblems.MultivariateProblems.objective(prob)
    x0 = prob.initial_x
end
@everywhere function fu(u,wait=1e-4) 
    sleep(wait)
    f(u)
end
@everywhere fu(u) = fu(u,1e-5)

r=[]
opts= Array{Array{Float64,1},1}()
[push!(opts,[]) for i in 1:(wrkrs+1)]
push!(r,@benchmarkable push!(opts[1],Optim.minimum(Optim.optimize(fu, x0.-100., x0.+100.0, x0, Optim.SAMIN(ns=20), Optim.Options(iterations=iters)))))
for i in 1:8
    a= @benchmarkable push!(opts[$i+1],Optim.minimum(Optim.optimize(fu, x0.-100., x0.+100.0, x0, Optim.SAMIN(ns=20,workers = workers()[1:$i]), Optim.Options(iterations=iters))))
    push!(r,a)
end
aa= BenchmarkTools.run.(r,samples=samples)


plot(
    0:(wrkrs),
    [minimum(i.times) for i in aa]./mean(aa[1].times),
    title = "Distributed-SAMIN",
    label= "rel. runtime",
    lw = 1,
    ylabel="rel. runtime",
    xlabel = "Workers",
    marker = (:dot, 2, 0.6),

    ribbon= [std(i.times) for i in aa]./mean(aa[1].times),
    )
plot!(0:(wrkrs), NaN.*(1:10), label = "f_min", marker = (:dot,:green, 3, 0.6),linecolor=:green, grid=false, legend=:right)
p=plot!(twinx(),0:(wrkrs), ylabel="f_min", [mean(i) for i in opts],yerror= [std(i) for i in opts], legend=false, marker = (:dot,:green, 3, 0.6),linecolor=:green)

savefig(p,"~/p.svg")
=#

