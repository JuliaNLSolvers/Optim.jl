## Dealing with constant parameters
In many applications, there may be factors that are relevant to the function evaluations,
but are fixed throughout the optimization. An obvious example is using data in a
likelihood function, but it could also be parameters we wish to hold constant.

Consider a squared error loss function that depends on some data `x` and `y`,
and parameters `betas`. As far as the solver is concerned, there should only be one
input argument to the function we want to minimize, call it `sqerror`.

The problem is that we want to optimize a function `sqerror` that really depends
on three inputs, and two of them are constant throught the optimization procedure.
To do this, we need to define the variables `x` and `y`
```jl
x = [1.0, 2.0, 3.0]
y = 1.0 + 2.0 * x + [-0.3, 0.3, -0.1]
```
We then simply define a function in three variables
```julia
function sqerror(betas, X, Y)
    err = 0.0
    for i in 1:length(X)
        pred_i = betas[1] + betas[2] * X[i]
        err += (Y[i] - pred_i)^2
    end
    return err
end
```
and then optimize the following anonymous function
```jl
res = optimize(b -> sqerror(b, x, y), [0.0, 0.0])
```
Alternatively, we can define a closure `sqerror(betas)` that is aware of the variables we
just defined
```jl
function sqerror(betas)
    err = 0.0
    for i in 1:length(x)
        pred_i = betas[1] + betas[2] * x[i]
        err += (y[i] - pred_i)^2
    end
    return err
end
```
We can then optimize the `sqerror` function just like any other function
```jl
res = optimize(sqerror, [0.0, 0.0])
```

## Avoid repeating computations
Say you are optimizing a function
```julia
f(x) = x[1]^2+x[2]^2
g!(x, stor) = [2x[1], 2x[2]]
```
In this situation, no calculations from `f` could be reused in `g!`. However, sometimes
there is a substantial similarity between the objective function, and gradient, and
some calculations can be reused.
The trick here is essentially the same as above. We use a closure or an anonymous function.
Basically, we define
```julia
function calculate_common!(x, last_x, buffer)
    if x != last_x
        copy!(last_x, x)
        #do whatever common calculations and save to buffer
    end
end

function f(x, buffer, last_x)
    calculate_common!(x, last_x, buffer)
    f_body # depends on buffer
end

function g!(x, stor, buffer, last_x)
    calculate_common!(x, last_x, buffer)
    g_body! # depends on buffer
end
```
and then the following
```julia
using Optim
x0 = ...
buffer = Array(...) # Preallocate an appropriate buffer
last_x = similar(x0)
df = TwiceDifferentiableFunction(x -> f(x, buffer, x0),
                                (x, stor) -> g!(x, stor, buffer, last_x))
optimize(df, x0)
```
## Provide gradients
As mentioned in the general introduction, passing analytical gradients can have an
impact on performance. To show an example of this, consider the separable extension of the
Rosenbrock function in dimension 5000, see [SROSENBR](ftp://ftp.numerical.rl.ac.uk/pub/cutest/sif/SROSENBR.SIF) in CUTEst.

Below, we use the gradients and objective functions from [mastsif](http://www.cuter.rl.ac.uk/Problems/mastsif.shtml) through [CUTEst.jl](https://github.com/JuliaOptimizers/CUTEst.jl).
We only show the first five iterations of an attempt to minimize the function using
Gradient Descent.
```jlcon
julia> @time optimize(f, x0, GradientDescent(),
                      OptimizationOptions(show_trace=true, iterations = 5))
Iter     Function value   Gradient norm
     0     4.850000e+04     2.116000e+02
     1     1.018734e+03     2.704951e+01
     2     3.468449e+00     5.721261e-01
     3     2.966899e+00     2.638790e-02
     4     2.511859e+00     5.237768e-01
     5     2.107853e+00     1.020287e-01
 21.731129 seconds (1.61 M allocations: 63.434 MB, 0.03% gc time)
Results of Optimization Algorithm
 * Algorithm: Gradient Descent
 * Starting Point: [1.2,1.0, ...]
 * Minimizer: [1.0287767703731154,1.058769439356144, ...]
 * Minimum: 2.107853e+00
 * Iterations: 5
 * Convergence: false
   * |x - x'| < 1.0e-32: false
   * |f(x) - f(x')| / |f(x)| < 1.0e-32: false
   * |g(x)| < 1.0e-08: false
   * Reached Maximum Number of Iterations: true
 * Objective Function Calls: 23
 * Gradient Calls: 23

julia> @time optimize(f, g!, x0, GradientDescent(),
                      OptimizationOptions(show_trace=true, iterations = 5))
Iter     Function value   Gradient norm
     0     4.850000e+04     2.116000e+02
     1     1.018769e+03     2.704998e+01
     2     3.468488e+00     5.721481e-01
     3     2.966900e+00     2.638792e-02
     4     2.511828e+00     5.237919e-01
     5     2.107802e+00     1.020415e-01
  0.009889 seconds (915 allocations: 270.266 KB)
Results of Optimization Algorithm
 * Algorithm: Gradient Descent
 * Starting Point: [1.2,1.0, ...]
 * Minimizer: [1.0287763814102757,1.05876866832087, ...]
 * Minimum: 2.107802e+00
 * Iterations: 5
 * Convergence: false
   * |x - x'| < 1.0e-32: false
   * |f(x) - f(x')| / |f(x)| < 1.0e-32: false
   * |g(x)| < 1.0e-08: false
   * Reached Maximum Number of Iterations: true
 * Objective Function Calls: 23
 * Gradient Calls: 23
```
The objective has obtained a value that is very similar between the two runs, but
the run with the analytical gradient is way faster.  It is possible that the finite
differences code can be improved, but generally the optimization will be slowed down
by all the function evaluations required to do the central finite differences calculations.
