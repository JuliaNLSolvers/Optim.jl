using Optim

# Accerlerated GD

dep_fagd(x) = x[1]^4
function dep_gagd!(x, storage)
    storage[1] = 4 * x[1]^3
    return
end

d = DifferentiableFunction(dep_fagd, dep_gagd!)

initial_x = [1.0]

Optim.accelerated_gradient_descent(d, initial_x, show_trace = true, iterations = 10)
Optim.momentum_gradient_descent(d, initial_x)

# GD

function dep_fgd(x)
  (x[1] - 5.0)^2
end

function dep_ggd!(x, storage)
  storage[1] = 2.0 * (x[1] - 5.0)
end

initial_x = [0.0]

d = DifferentiableFunction(dep_fgd, dep_ggd!)

results = Optim.gradient_descent(d, initial_x)
@assert isempty(results.trace.states)
@assert results.gr_converged
@assert norm(results.minimum - [5.0]) < 0.01

# BFGS

function dep_dbfgs(x)
  x[1]^2 + (2.0 - x[2])^2
end
function dep_gbfgs!(x, storage)
  storage[1] = 2.0 * x[1]
  storage[2] = -2.0 * (2.0 - x[2])
end
d2 = DifferentiableFunction(dep_dbfgs, dep_gbfgs!)
initial_x = [100.0, 100.0]

results = Optim.bfgs(d2, initial_x)
@assert length(results.trace.states) == 0
@assert results.gr_converged
@assert norm(results.minimum - [0.0, 2.0]) < 0.01

d2 = Optim.autodiff(f2, Float64, 2)
results = Optim.bfgs(d2, initial_x)
@assert length(results.trace.states) == 0
@assert results.gr_converged
@assert norm(results.minimum - [0.0, 2.0]) < 0.01

# Brent

dep_fbrent(x) = 2x^2+3x+1

results = Optim.brent(dep_fbrent, -2.0, 1.0)

@assert results.converged
@assert abs(results.minimum+0.75) < 1e-7

# Golden Section

results = Optim.golden_section(dep_fbrent, -2.0, 1.0)

@assert results.converged
@assert abs(results.minimum+0.75) < 1e-7

# CG

dep_objective(X, B) = sum((X.-B).^2)/2

function dep_objective_gradient!(X, G, B)
    for i = 1:length(G)
        G[i] = X[i]-B[i]
    end
end

srand(1)
B = rand(2,2)
df = Optim.DifferentiableFunction(X -> dep_objective(X, B), (X, G) -> dep_objective_gradient!(X, G, B))
results = Optim.cg(df, rand(2,2))
@assert Optim.converged(results)
@assert results.f_minimum < 1e-8

# LBFGS

function dep_flbfgs(x::Vector)
  (309.0 - 5.0 * x[1])^2 + (17.0 - x[2])^2
end
function dep_glbfgs!(x::Vector, storage::Vector)
  storage[1] = -10.0 * (309.0 - 5.0 * x[1])
  storage[2] = -2.0 * (17.0 - x[2])
end

d = DifferentiableFunction(dep_flbfgs, dep_glbfgs!)

initial_x = [10.0, 10.0]
m = 10
store_trace, show_trace = false, false

results = Optim.l_bfgs(d, initial_x)
@assert length(results.trace.states) == 0
@assert results.gr_converged
@assert norm(results.minimum - [309.0 / 5.0, 17.0]) < 0.01

# Nelder Mead

function dep_f_nm(x::Vector)
  (100.0 - x[1])^2 + x[2]^2
end

initial_x = [0.0, 0.0]

results = Optim.nelder_mead(dep_f_nm, initial_x)

@assert results.f_converged
@assert norm(results.minimum - [100.0, 0.0]) < 0.01
@assert length(results.trace.states) == 0

# Newton

function dep_f1(x::Vector)
    (x[1] - 5.0)^4
end

function dep_g1!(x::Vector, storage::Vector)
    storage[1] = 4.0 * (x[1] - 5.0)^3
end

function dep_h1!(x::Vector, storage::Matrix)
    storage[1, 1] = 12.0 * (x[1] - 5.0)^2
end

d = TwiceDifferentiableFunction(dep_f1, dep_g1!, dep_h1!)

results = Optim.newton(d, [0.0])
@assert length(results.trace.states) == 0
@assert results.gr_converged
@assert norm(results.minimum - [5.0]) < 0.01

# Simulated Annealing

function f_s(x::Vector)
    (x[1] - 5.0)^4
end

results = Optim.simulated_annealing(f_s, [0.0])
@assert norm(results.minimum - [5.0]) < 0.1

