# Preconditioning

The `GradientDescent`, `ConjugateGradient` and `LBFGS` methods support preconditioning. A preconditioner
can be thought of as a change of coordinates under which the Hessian is better conditioned. With a
good preconditioner substantially improved convergence is possible.

A preconditioner `P`can be of any type as long as the following two methods are
implemented:

* `ldiv!(pgr, P, gr)` : apply `P` to a vector `gr` and store in `pgr`
      (intuitively, `pgr = P \ gr`)
* `dot(x, P, y)` : the inner product induced by `P`
      (intuitively, `dot(x, P * y)`)

Precisely what these operations mean, depends on how `P` is stored. Commonly, we store a matrix `P` which
approximates the Hessian (not the inverse Hessian) in some vague sense.

Finally, it is possible to update the preconditioner as the state variable `x`
changes. This is done through  `precondprep` which is passed to the
optimizers as kw-argument, e.g.,
```jl
   method=ConjugateGradient(P = precond(100), precondprep = (P, x)->copyto!(P, precond(x)))
```
though in this case it would always return the same matrix.

!!! note
    Preconditioning is also used in `Fminbox` even if the user does not provide a preconditioner. This is because we have a barrier term causing the problem to be ill-conditioned. The preconditioner uses the hessian of the barrier term to improve the conditioning.

## Example
Below, we see an example where a function is minimized without and with a preconditioner
applied.
```jl
using ForwardDiff, Optim, SparseArrays
plap(U; n = length(U)) = (n - 1) * sum((0.1 .+ diff(U) .^ 2) .^ 2) - sum(U) / (n - 1)
plap1(U; n = length(U), dU = diff(U), dW = 4 .* (0.1 .+ dU .^ 2) .* dU) =
      (n - 1) .* ([0.0; dW] .- [dW; 0.0]) .- ones(n) / (n - 1)
precond(x::Vector) = precond(length(x))
precond(n::Number) = spdiagm(-1 => -ones(n - 1), 0 => 2 * ones(n), 1 => -ones(n - 1)) * (n + 1)
f(X) = plap([0; X; 0])
g!(G, X) = copyto!(G, (plap1([0; X; 0]))[2:end-1])
initial_x = zeros(100)

result = Optim.optimize(f, g!, initial_x, method = ConjugateGradient(P = nothing))
result = Optim.optimize(f, g!, initial_x, method = ConjugateGradient(P = precond(initial_x)))
```
The former optimize call converges at a slower rate than the latter. Looking at a
 plot of the 2D version of the function shows the problem.

![plap](./plap.png)

The contours are shaped like ellipsoids, but we would rather want them to be circles.
Using the preconditioner effectively changes the coordinates such that the contours
becomes less ellipsoid-like. Benchmarking shows that using preconditioning provides
 an approximate speed-up factor of 15 in this 100 dimensional case.


## References
