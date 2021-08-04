# Hessian-Free Newton's Method With a Trust Region
## Constructor
```julia
KrylovTrustRegion(; initial_radius = 1.0,
                    max_radius = 100.0,
                    eta = 0.1,
                    rho_lower = 0.25,
                    rho_upper = 0.75,
                    cg_tol = 0.01)
```

The constructor takes keywords that determine the initial and maximal size of the trust region, when to grow and shrink the region, and how close the function should be to the quadratic approximation.  The notation follows chapter four of Numerical Optimization.  Below, ```rho``` ``=\rho`` refers to the ratio of the actual function change to the change in the quadratic approximation for a given step. ```cg_tol``` ...

* `initial_radius:`The starting trust region radius
*  `max_radius:` The largest allowable trust region radius
*  `eta:` When ```rho``` is at least ```eta```, accept the step.
*  `rho_lower: ` When ```rho``` is less than ```rho_lower```, shrink the trust region.
*  `rho_upper:` When ```rho``` is greater than ```rho_upper```, grow the trust region (though no greater than ```delta_hat```).
*  `cg_tol:` Tolerance on ...

## Description
Newton's method with a trust region is designed to take advantage of the second-order information in a function's Hessian, but with more stability than Newton's method when functions are not globally well-approximated by a quadratic.  This is achieved by repeatedly minimizing quadratic approximations within a dynamically-sized "trust region" in which the function is assumed to be locally quadratic [1].

Newton's method optimizes a quadratic approximation to a function. When a function is well approximated by a quadratic (for example, near an optimum), Newton's method converges very quickly by exploiting the second-order information in the Hessian matrix.  However, when the function is not well-approximated by a quadratic, either because the starting point is far from the optimum or the function has a more irregular shape, Newton steps can be erratically large, leading to distant, irrelevant areas of the space.

The standard Newton method requires repeatedly calculating the Hessian of the optimization objective to solve a linear system, which for large problems can require a lot of time and memory. A completely different way to solve large linear systems is to use a Krylov subpsace method [2]. This method does not require the calculation of the full Hessian, instead it uses Hessian vector products to reduce memory and time requirements.
## Example

```julia
using Optim, OptimTestProblems
prob = UnconstrainedProblems.examples["Rosenbrock"];
function rosenbrock_Hv!(storage::Vector,x::Vector, v::Vector)
    storage[1] = (2.0 - 400.0 * x[2] + 1200.0 * x[1]^2)*v[1] + (-400.0 * x[1])v[2]
    storage[2] = (-400.0 * x[1])v[1] + (200)v[2]
end
function rosenbrock_f_g!(storage::Vector,x::Vector)
    prob.g!(storage,x)
    prob.f(x)
end
ddf = Optim.TwiceDifferentiableHV(prob.f, rosenbrock_f_g!, rosenbrock_Hv!, prob.initial_x)
res = Optim.optimize(ddf,prob.initial_x, Optim.KrylovTrustRegion())
```
## References

[1] Nocedal, Jorge, and Stephen Wright. Numerical optimization. Springer Science & Business Media, 2006.
[2] Knoll, Dana, and Keyes David. acobian-free Newton-Krylov methods: a survey of approaches and applications. Journal of Computational Physics , 2004. 
