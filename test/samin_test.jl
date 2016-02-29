include("samin.jl")

junk=2. # shows use of opj. fun. as a closure
function sse(x)
    objvalue = junk + sum(x.*x)
end

k = 5
x = rand(k,1)
lb = -ones(k,1)
ub = -lb

# converge to global opt
println("normal convergence, terse output")
@time xopt = samin(sse, x, lb, ub, verbosity=1)
# converge to global opt, see final parameters
println("normal convergence, more verbose output")
@time xopt = samin(sse, x, lb, ub, verbosity=2)
# no convergence within iteration limit
println("no convergence within iter limit")
@time xopt = samin(sse, x, lb, ub, maxevals=10, verbosity=1)
# initial point out of bounds
println("initial point out of bounds")
lb = 0.5*ub
x[1,1] = 0.2
xopt = samin(sse, x, lb, ub, verbosity=1)
# optimum on bound of parameter space
println("optimum on bounds of parameter space")
x = 0.5 .+ 0.5*rand(k,1)
xopt = samin(sse, x, lb, ub, verbosity=1)
;
