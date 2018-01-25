# Complex optimization
Optimization of functions defined on complex inputs (``\mathbb{C}^n
\to \mathbb{R}``) is supported by simply passing a complex ``x`` as
input. All zeroth and first order optimization algorithms are
supported.

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
seen as a function of R^2n to R^2n, it might not be
complex-differentiable. For instance, take ``f(z) = \mbox{Re}(z)^2``.
Then ``g(z) = 2 \mbox{Re}(z)``, which is not complex-differentiable
(holomorphic). Therefore,
the Hessian of a ``\mathbb{C}^n \to \mathbb{R}`` function is in
general not well-defined as a ``n \times n`` complex matrix (only as a
``2n \times 2n`` real matrix), and therefore
second-order optimization algorithms are not applicable directly. To
use second-order optimization, convert to real variables.


## Examples
**TODO**


## References
**TODO**
