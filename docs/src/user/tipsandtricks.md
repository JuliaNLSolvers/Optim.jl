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
initial_x = ...
buffer = Array{eltype(initial_x)}(...) # Preallocate an appropriate buffer
last_x = similar(initial_x)
df = TwiceDifferentiableFunction(x -> f(x, buffer, initial_x),
                                (x, stor) -> g!(x, stor, buffer, last_x))
optimize(df, initial_x)
```
## Provide gradients
As mentioned in the general introduction, passing analytical gradients can have an
impact on performance. To show an example of this, consider the separable extension of the
Rosenbrock function in dimension 5000, see [SROSENBR](ftp://ftp.numerical.rl.ac.uk/pub/cutest/sif/SROSENBR.SIF) in CUTEst.

Below, we use the gradients and objective functions from [mastsif](http://www.cuter.rl.ac.uk/Problems/mastsif.shtml) through [CUTEst.jl](https://github.com/JuliaOptimizers/CUTEst.jl).
We only show the first five iterations of an attempt to minimize the function using
Gradient Descent.
```jlcon
julia> @time optimize(f, initial_x, GradientDescent(),
                      Optim.Options(show_trace=true, iterations = 5))
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

julia> @time optimize(f, g!, initial_x, GradientDescent(),
                      Optim.Options(show_trace=true, iterations = 5))
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

## Early stopping
Sometimes it might be of interest to stop the optimizer early. The simplest way to
do this is to set the `iterations` keyword in `Optim.Options` to some number.
This will prevent the iteration counter exceeding some limit, with the standard value
being 1000. Alternatively, it is possible to put a soft limit on the run time of
the optimization procedure by setting the `time_limit` keyword in the `Optim.Options`
constructor.
```julia
using Optim
problem = Optim.UnconstrainedProblems.examples["Rosenbrock"]

f = problem.f
initial_x = problem.initial_x

function slow(x)
    sleep(0.1)
    f(x)
end

start_time = time()

optimize(slow, zeros(2), NelderMead(), Optim.Options(time_limit = 3.0))
```
This will stop after about three seconds. If it is more important that we stop before the limit
is reached, it is possible to use a callback with a simple model for predicting how much
time will have passed when the next iteration is over. Consider the following code
```julia
using Optim
problem = Optim.UnconstrainedProblems.examples["Rosenbrock"]

f = problem.f
initial_x = problem.initial_x

function very_slow(x)
    sleep(.5)
    f(x)
end

start_time = time()
time_to_setup = zeros(1)
function advanced_time_control(x)
    println(" * Iteration:       ", x.iteration)
    so_far =  time()-start_time
    println(" * Time so far:     ", so_far)
    if x.iteration == 0
        time_to_setup[:] = time()-start_time
    else
        expected_next_time = so_far + (time()-start_time-time_to_setup[1])/(x.iteration)
        println(" * Next iteration ≈ ", expected_next_time)
        println()
        return expected_next_time < 13 ? false : true
    end
    println()
    false
end
optimize(very_slow, zeros(2), NelderMead(), Optim.Options(callback = advanced_time_control))
```
It will try to predict the elapsed time after the next iteration is over, and stop now
if it is expected to exceed the limit of 13 seconds. Running it, we get something like
the following output
```jlcon
julia> optimize(very_slow, zeros(2), NelderMead(), Optim.Options(callback = advanced_time_control))
 * Iteration:       0
 * Time so far:     2.219298839569092

 * Iteration:       1
 * Time so far:     3.4006409645080566
 * Next iteration ≈ 4.5429909229278564

 * Iteration:       2
 * Time so far:     4.403923988342285
 * Next iteration ≈ 5.476739525794983

 * Iteration:       3
 * Time so far:     5.407265901565552
 * Next iteration ≈ 6.4569235642751055

 * Iteration:       4
 * Time so far:     5.909044027328491
 * Next iteration ≈ 6.821732044219971

 * Iteration:       5
 * Time so far:     6.912338972091675
 * Next iteration ≈ 7.843148183822632

 * Iteration:       6
 * Time so far:     7.9156060218811035
 * Next iteration ≈ 8.85849153995514

 * Iteration:       7
 * Time so far:     8.918903827667236
 * Next iteration ≈ 9.870419979095459

 * Iteration:       8
 * Time so far:     9.922197818756104
 * Next iteration ≈ 10.880185931921005

 * Iteration:       9
 * Time so far:     10.925468921661377
 * Next iteration ≈ 11.888488478130764

 * Iteration:       10
 * Time so far:     11.92870283126831
 * Next iteration ≈ 12.895747828483582

 * Iteration:       11
 * Time so far:     12.932114839553833
 * Next iteration ≈ 13.902462200684981

Results of Optimization Algorithm
 * Algorithm: Nelder-Mead
 * Starting Point: [0.0,0.0]
 * Minimizer: [0.23359374999999996,0.042187499999999996, ...]
 * Minimum: 6.291677e-01
 * Iterations: 11
 * Convergence: false
   *  √(Σ(yᵢ-ȳ)²)/n < 1.0e-08: false
   * Reached Maximum Number of Iterations: false
 * Objective Function Calls: 24
```
