## Gradients and Hessians
To use first- and second-order methods, you need to provide gradients and Hessians, either in-place or out-of-place. There are three main ways of specifying derivatives: analytic, finite-difference and automatic differentiation.

## Analytic
This results in the fastest run times, but requires the user to perform the often tedious task of computing the derivatives by hand. The gradient of complicated objective functions (e.g. involving the solution of algebraic equations, differential equations, eigendecompositions, etc.) can be computed efficiently using the adjoint method (see e.g. [these lecture notes](https://math.mit.edu/~stevenj/18.336/adjoint.pdf)). In particular, assuming infinite memory, the gradient of a ``\mathbb{R}^N \to \mathbb{R}`` function ``f`` can always be computed with a runtime comparable with only one evaluation of ``f``, no matter how large ``N``.

To use analytic derivatives, simply pass `g!` and `h!` functions to `optimize`.

## Finite differences
This uses the functionality in [DiffEqDiffTools.jl](https://github.com/JuliaDiffEq/DiffEqDiffTools.jl) to compute gradients and Hessians through central finite differences: ``f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}``. For a ``\mathbb{R}^N \to \mathbb{R}`` objective function ``f``, this requires ``2N`` evaluations of ``f``. It is therefore efficient in low dimensions but slow when ``N`` is large. It is also inaccurate: ``h`` is chosen equal to ``\epsilon^{1/3}`` where ``\epsilon`` is the machine epsilon (about ``10^{-16}`` for `Float64`) to balance the truncation and rounding errors, resulting in an error of ``\epsilon^{2/3}`` (about ``10^{-11}`` for `Float64`) for the derivative.

Finite differences are on by default if gradients and Hessians are not supplied to the `optimize` call.

## Automatic differentiation
Automatic differentiation techniques are a middle ground between finite differences and analytic computations. They are exact up to machine precision, and do not require intervention from the user. They come in two main flavors: [forward and reverse mode](https://en.wikipedia.org/wiki/Automatic_differentiation). Forward-mode automatic differentiation is relatively straightforward to implement by propagating the sensitivities of the input variables, and is often faster than finite differences. The disadvantage is that the objective function has to be written using only Julia code. Forward-mode automatic differentiation still requires a runtime comparable to ``N`` evaluations of ``f``, and is therefore costly in large dimensions, like finite differences.

Reverse-mode automatic differentiation can be seen as an automatic implementation of the adjoint method mentioned above, and requires a runtime comparable to only one evaluation of ``f``. It is however considerably more complex to implement, requiring to record the execution of the program to then run it backwards, and incurs a larger overhead.

Forward-mode automatic differentiation is supported through the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) package by providing the `autodiff=:forward` keyword to `optimize`. Reverse-mode automatic differentiation is not supported explicitly yet (although you can use it by writing your own `g!` function). There are a number of implementations in Julia, such as [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl).

## Example

Let us consider the Rosenbrock example again.
```julia
function f(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end

initial_x = zeros(2)
```
Let us see if BFGS and Newton's Method can solve this problem with the functions
provided.
```jlcon
julia> Optim.minimizer(optimize(f, g!, h!, initial_x, BFGS()))
2-element Array{Float64,1}:
 1.0
 1.0

julia> Optim.minimizer(optimize(f, g!, h!, initial_x, Newton()))

2-element Array{Float64,1}:
 1.0
 1.0
```
This is indeed the case. Now let us use finite differences for BFGS.
```jlcon
julia> Optim.minimizer(optimize(f, initial_x, BFGS()))
2-element Array{Float64,1}:
 1.0
 1.0
```
Still looks good. Returning to automatic differentiation, let us try both solvers using this
method.  We enable [forward mode](https://github.com/JuliaDiff/ForwardDiff.jl) automatic
differentiation by using the `autodiff = :forward` keyword.
```jlcon
julia> Optim.minimizer(optimize(f, initial_x, BFGS(); autodiff = :forward))
2-element Array{Float64,1}:
 1.0
 1.0

julia> Optim.minimizer(optimize(f, initial_x, Newton(); autodiff = :forward))
2-element Array{Float64,1}:
 1.0
 1.0
```
Indeed, the minimizer was found, without providing any gradients or Hessians.
