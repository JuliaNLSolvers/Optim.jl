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
