function f_lm(x)
  [x[1], 2.0 - x[2]]
end
function g_lm(x)
  [1.0 0.0; 0.0 -1.0]
end

initial_x = [100.0, 100.0]

results = Optim.levenberg_marquardt(f_lm, g_lm, initial_x)
@assert norm(Optim.minimizer(results) - [0.0, 2.0]) < 0.01


function rosenbrock_res(x, r)
    r[1] = 10.0 * (x[2] - x[1]^2 )
    r[2] =  1.0 - x[1]
    return r
end

function rosenbrock_jac(x, j)
    j[1, 1] = -20.0 * x[1]
    j[1, 2] =  10.0
    j[2, 1] =  -1.0
    j[2, 2] =   0.0
    return j
end

r = zeros(2)
j = zeros(2,2)

frb(x) = rosenbrock_res(x, r)
grb(x) = rosenbrock_jac(x, j)

initial_xrb = [-1.2, 1.0]

results = Optim.levenberg_marquardt(frb, grb, initial_xrb)

@assert norm(Optim.minimizer(results) - [1.0, 1.0]) < 0.01

# tests for #178, taken from LsqFit.jl, but stripped
let
    srand(12345)

    model(x, p) = p[1]*exp(-x.*p[2])

    xdata = linspace(0,10,20)
    ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))

    f_lsq = p -> p[1]*exp(-xdata.*p[2])-ydata
    g_lsq = Calculus.jacobian(f_lsq)
    results = Optim.levenberg_marquardt(f_lsq, g_lsq, [0.5, 0.5])

    @assert norm(Optim.minimizer(results) - [1.0, 2.0]) < 0.05
end
