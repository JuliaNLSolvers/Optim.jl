# Newton's Method With a Trust Region
## Constructor
```julia
NewtonTrustRegion(; initial_delta = 1.0,
                    delta_hat = 100.0,
                    eta = 0.1,
                    rho_lower = 0.25,
                    rho_upper = 0.75)
```

The constructor takes keywords that determine the initial and maximal size of the trust region, when to grow and shrink the region, and how close the function should be to the quadratic approximation.  The notation follows chapter four of Numerical Optimization.  Below, ```rho``` ``=\rho`` refers to the ratio of the actual function change to the change in the quadratic approximation for a given step.

* `initial_delta:`The starting trust region radius
*  `delta_hat:` The largest allowable trust region radius
*  `eta:` When ```rho``` is at least ```eta```, accept the step.
*  `rho_lower: ` When ```rho``` is less than ```rho_lower```, shrink the trust region.
*  `rho_upper:` When ```rho``` is greater than ```rho_upper```, grow the trust region (though no greater than ```delta_hat```).

## Description
Newton's method with a trust region is designed to take advantage of the second-order information in a function's Hessian, but with more stability than Newton's method when functions are not globally well-approximated by a quadratic.  This is achieved by repeatedly minimizing quadratic approximations within a dynamically-sized "trust region" in which the function is assumed to be locally quadratic [1].

Newton's method optimizes a quadratic approximation to a function.  When a function is well approximated by a quadratic (for example, near an optimum), Newton's method converges very quickly by exploiting the second-order information in the Hessian matrix.  However, when the function is not well-approximated by a quadratic, either because the starting point is far from the optimum or the function has a more irregular shape, Newton steps can be erratically large, leading to distant, irrelevant areas of the space.

Trust region methods use second-order information but restrict the steps to be within a "trust region" where the function is believed to be approximately quadratic.  At iteration ``k``, a trust region method chooses a step ``p`` to minimize a quadratic approximation to the objective such that the step size is no larger than a given trust region size, ``\Delta_k``.

```math
\underset{p\in\mathbb{R}^n}\min m_k(p) = f_k + g_k^T p + \frac{1}{2}p^T B_k p \quad\textrm{such that } ||p||\le \Delta_k
```

Here, ``p`` is the step to take at iteration ``k``, so that ``x_{k+1} = x_k + p``.   In the definition of ``m_k(p)``, ``f_k = f(x_k)`` is the value at the previous location, ``g_k=\nabla f(x_k)`` is the gradient at the previous location, ``B_k = \nabla^2 f(x_k)`` is the Hessian matrix at the previous iterate, and ``||\cdot||`` is the Euclidian norm.

If the trust region size, ``\Delta_k``, is large enough that the minimizer of the quadratic approximation ``m_k(p)`` has ``||p|| \le \Delta_k``, then the step is the same as an ordinary Newton step.  However, if the unconstrained quadratic minimizer lies outside the trust region, then the minimizer to the constrained problem will occur on the boundary, i.e. we will have ``||p|| = \Delta_k``.  It turns out that when the Cholesky decomposition of ``B_k`` can be computed, the optimal ``p`` can be found numerically with relative ease.  ([1], section 4.3)  This is the method currently used in Optim.

It makes sense to adapt the trust region size, ``\Delta_k``, as one moves through the space and assesses the quality of the quadratic fit.  This adaptation is controlled by the parameters ``\eta``, ``\rho_{lower}``, and ``\rho_{upper}``, which are parameters to the ```NewtonTrustRegion``` optimization method.  For each step, we calculate

```math
\rho_k := \frac{f(x_{k+1}) - f(x_k)}{m_k(p) - m_k(0)}
```

Intuitively, ``\rho_k`` measures the quality of the quadratic approximation: if ``\rho_k \approx 1``, then our quadratic approximation is reasonable.  If  ``p`` was on the boundary and ``\rho_k > \rho_{upper}``, then perhaps we can benefit from larger steps.  In this case, for the next iteration we grow the trust region geometrically up to a maximum of ``\hat\Delta``:

```math
\rho_k > \rho_{upper} \Rightarrow \Delta_{k+1} = \min(2 \Delta_k, \hat\Delta).
```

Conversely, if ``\rho_k < \rho_{lower}``, then we shrink the trust region geometrically:

``\rho_k < \rho_{lower} \Rightarrow \Delta_{k+1} = 0.25 \Delta_k``.
Finally, we only accept a point if its decrease is appreciable compared to the quadratic approximation.  Specifically, a step is only accepted ``\rho_k > \eta``.  As long as we choose ``\eta`` to be less than ``\rho_{lower}``, we will shrink the trust region whenever we reject a step.  Eventually, if the objective function is locally quadratic, ``\Delta_k`` will become small enough that a quadratic approximation will be accurate enough to make progress again.

## Example

```julia
using Optim, OptimTestProblems
prob = OptimTestProblems.UnconstrainedProblems.examples["Rosenbrock"];
res = Optim.optimize(prob.f, prob.g!, prob.h!, prob.initial_x, NewtonTrustRegion())
```

## References

[1] Nocedal, Jorge, and Stephen Wright. Numerical optimization. Springer Science & Business Media, 2006.
