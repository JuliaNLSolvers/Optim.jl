# Conjugate Gradient Descent
## Constructor
```julia
ConjugateGradient(; alphaguess = LineSearches.InitialHagerZhang(),
                    linesearch = LineSearches.HagerZhang(),
                    eta = 0.4,
                    P = nothing,
                    precondprep = (P, x) -> nothing)
```

## Description
The `ConjugateGradient` method implements Hager and Zhang (2006) and elements from
Hager and Zhang (2013). Notice, that the default `linesearch` is `HagerZhang` from
LineSearches.jl. This line search is exactly the one proposed in Hager and Zhang (2006).
The constant ``eta`` is used in determining the next step direction, and the default
here deviates from the one used in the original paper (``0.01``). It needs to be
a strictly positive number.

## Example
Let's optimize the 2D Rosenbrock function. The function and gradient are given by
```
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
function g!(storage, x)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end
```
we can then try to optimize this function from `x=[0.0, 0.0]`
```
julia> optimize(f, g!, zeros(2), ConjugateGradient())
Results of Optimization Algorithm
 * Algorithm: Conjugate Gradient
 * Starting Point: [0.0,0.0]
 * Minimizer: [1.000000002262018,1.0000000045408348]
 * Minimum: 5.144946e-18
 * Iterations: 21
 * Convergence: true
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 2.09e-10
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = 1.55e+00 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 3.36e-09
   * stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 54
 * Gradient Calls: 39
```
We can compare this to the default first order solver in Optim.jl
```
 julia> optimize(f, g!, zeros(2))

 Results of Optimization Algorithm
  * Algorithm: L-BFGS
  * Starting Point: [0.0,0.0]
  * Minimizer: [0.9999999999373614,0.999999999868622]
  * Minimum: 7.645684e-21
  * Iterations: 16
  * Convergence: true
    * |x - x'| ≤ 0.0e+00: false
      |x - x'| = 3.48e-07
    * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
      |f(x) - f(x')| = 9.03e+06 |f(x)|
    * |g(x)| ≤ 1.0e-08: true
      |g(x)| = 2.32e-09
    * stopped by an increasing objective: false
    * Reached Maximum Number of Iterations: false
  * Objective Calls: 53
  * Gradient Calls: 53
```
We see that for this objective and starting point, `ConjugateGradient()` requires
fewer gradient evaluations to reach convergence.

## References
- W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent. ACM Transactions on Mathematical Software 32: 113-137.
- W. W. Hager and H. Zhang (2013), The Limited Memory Conjugate Gradient Method. SIAM Journal on Optimization, 23, pp. 2150-2168.
