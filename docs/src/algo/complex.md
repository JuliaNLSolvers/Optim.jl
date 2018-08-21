# Complex optimization
Optimization of functions defined on complex inputs (``\mathbb{C}^n
\to \mathbb{R}``) is supported by simply passing a complex ``x`` as
input. The algorithms supported are all those which can naturally be
extended to work with complex numbers: simulated annealing and all the
first-order methods.

The gradient of a complex-to-real function is defined as the only
vector ``g`` such that
```math
f(x+h) = f(x) + \mbox{Re}(g' * h) + \mathcal{O}(h^2).
```
This is sometimes written
```math
g = \frac{df}{d(z*)} = \frac{df}{d(\mbox{Re}(z))} + i \frac{df}{d(\mbox{Im(z)})}.
```

The gradient of a ``\mathbb{C}^n \to \mathbb{R}`` function is a
``\mathbb{C}^n \to \mathbb{C}^n`` map. Even if it is differentiable when
seen as a function of ``\mathbb{R}^{2n}`` to ``\mathbb{R}^{2n}``, it
might not be
complex-differentiable. For instance, take ``f(z) = \mbox{Re}(z)^2``.
Then ``g(z) = 2 \mbox{Re}(z)``, which is not complex-differentiable
(holomorphic). Therefore,
the Hessian of a ``\mathbb{C}^n \to \mathbb{R}`` function is in
general not well-defined as a ``n \times n`` complex matrix (only as a
``2n \times 2n`` real matrix), and therefore
second-order optimization algorithms are not applicable directly. To
use second-order optimization, convert to real variables.


## Examples
We show how to minimize a quadratic plus quartic function with
the `LBFGS` optimization algorithm.

```jl
using Random
Random.seed!(0) # Set the seed for reproducibility
# μ is the strength of the quartic. μ = 0 is just a quadratic problem
n = 4
A = randn(n,n) + im*randn(n,n)
A = A'A + I
b = randn(n) + im*randn(n)
μ = 1.0

fcomplex(x) = real(dot(x,A*x)/2 - dot(b,x)) + μ*sum(abs.(x).^4)
gcomplex(x) = A*x-b + 4μ*(abs.(x).^2).*x
gcomplex!(stor,x) = copyto!(stor,gcomplex(x))

x0 = randn(n)+im*randn(n)

res = optimize(fcomplex, gcomplex!, x0, LBFGS())
```

The output of the optimization is
```
Results of Optimization Algorithm
 * Algorithm: L-BFGS
 * Starting Point: [0.48155603952425174 - 1.477880724921868im,-0.3219431528959694 - 0.18542418173298963im, ...]
 * Minimizer: [0.14163543901272568 - 0.034929496785515886im,-0.1208600058040362 - 0.6125620908171383im, ...]
 * Minimum: -1.568997e+00
 * Iterations: 16
 * Convergence: true
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 3.28e-09
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = -4.25e-16 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 6.33e-11
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 48
 * Gradient Calls: 48
```

Similarly, with `ConjugateGradient`.

``` julia
res = optimize(fcomplex, gcomplex!, x0, ConjugateGradient())
```

```
Results of Optimization Algorithm
 * Algorithm: Conjugate Gradient
 * Starting Point: [0.48155603952425174 - 1.477880724921868im,-0.3219431528959694 - 0.18542418173298963im, ...]
 * Minimizer: [0.1416354378490425 - 0.034929499492595516im,-0.12086000949769983 - 0.6125620892675705im, ...]
 * Minimum: -1.568997e+00
 * Iterations: 23
 * Convergence: false
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 8.54e-10
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = -4.25e-16 |f(x)|
   * |g(x)| ≤ 1.0e-08: false
     |g(x)| = 3.72e-08
   * Stopped by an increasing objective: true
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 51
 * Gradient Calls: 29
```

### Differentation
The finite difference methods used by `Optim` support real functions
with complex inputs.

``` julia
res = optimize(fcomplex, x0, LBFGS())
```

```
Results of Optimization Algorithm
 * Algorithm: L-BFGS
 * Starting Point: [0.48155603952425174 - 1.477880724921868im,-0.3219431528959694 - 0.18542418173298963im, ...]
 * Minimizer: [0.1416354390108624 - 0.034929496786122484im,-0.12086000580073922 - 0.6125620908025359im, ...]
 * Minimum: -1.568997e+00
 * Iterations: 16
 * Convergence: true
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 3.28e-09
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: true
     |f(x) - f(x')| = 0.00e+00 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 1.04e-10
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 48
 * Gradient Calls: 48
```

Automatic differentiation support for complex inputs may come when
[Cassete.jl](https://github.com/JuliaDiff/Capstan.jl) is ready.

## References

 - Sorber, L., Barel, M. V., & Lathauwer, L. D. (2012). Unconstrained optimization of real functions in complex variables. SIAM Journal on Optimization, 22(3), 879-898.
 - Kreutz-Delgado, K. (2009). The complex gradient operator and the CR-calculus. arXiv preprint arXiv:0906.4835.
