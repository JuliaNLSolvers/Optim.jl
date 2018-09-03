## Algorithm choice

There are two main settings you must choose in Optim: the algorithm and the linesearch.

## Algorithms

The first choice to be made is that of the order of the method. Zeroth-order methods do not have gradient information, and are very slow to converge, especially in high dimension. First-order methods do not have access to curvature information and can take a large number of iterations to converge for badly conditioned problems. Second-order methods can converge very quickly once in the vicinity of a minimizer. Of course, this enhanced performance comes at a cost: the objective function has to be differentiable, you have to supply gradients and Hessians, and, for second order methods, a linear system has to be solved at each step.

If you can provide analytic gradients and Hessians, and the dimension of the problem is not too large, then second order methods are very efficient. The Newton method with trust region is the method of choice. 

When you do not have an explicit Hessian or when the dimension becomes large enough that the linear solve in the Newton method becomes the bottleneck, first order methods should be preferred. BFGS is a very efficient method, but also requires a linear system solve. LBFGS usually has a performance very close to that of BFGS, and avoids linear system solves (the parameter `m` can be tweaked: increasing it can improve the convergence, at the expense of memory and time spent in linear algebra operations). The conjugate gradient method usually converges less quickly than LBFGS, but requires less memory. Gradient descent should only be used for testing. Acceleration methods are experimental.

When the objective function is non-differentiable or you do not want to use gradients, use zeroth-order methods. Nelder-Mead is currently the most robust.

## Linesearches

Linesearches are used in every first- and second-order method except for the trust-region Newton method. Linesearch routines attempt to locate quickly an approximate minimizer of the univariate function ``\alpha \to f(x+ \alpha d)``, where ``d`` is the descent direction computed by the algorithm. They vary in how accurate this minimization is. Two good linesearches are BackTracking and HagerZhang, the former being less stringent than the latter. For well-conditioned objective functions and methods where the step is usually well-scaled (such as LBFGS or Newton), a rough linesearch such as BackTracking is usually the most performant. For badly behaved problems or when extreme accuracy is needed (gradients below the square root of the machine epsilon, about ``10^{-8}`` with `Float64`), the HagerZhang method proves more robust. An exception is the conjugate gradient method which requires an accurate linesearch to be efficient, and should be used with the HagerZhang linesearch.

## Summary

As a very crude heuristic:

For a low-dimensional problem with analytic gradients and Hessians, use the Newton method with trust region. For larger problems or when there is no analytic Hessian, use LBFGS, and tweak the parameter `m` if needed. If the function is non-differentiable, use Nelder-Mead. Use the HagerZhang linesearch for robustness and BackTracking for speed.
