# Optimization Functions for Julia

### Usage Examples

If you're just getting started, you probably want to use `optimize()`, which wraps the specific algorithms currently implemented and selects a good one based on the amount of information you can provide. See the examples below:

    load("src/init.jl")

    eta = 0.9

    function f(x)
      (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
    end

    function g(x)
      [x[1], eta * x[2]]
    end

    function h(x)
      [1.0 0.0; 0.0 eta]
    end

    # If you don't have gradient, uses Nelder-Mead.
    results = optimize(f, [127.0, 921.0])
    @assert norm(results.minimum - [0.0, 0.0]) < 0.01

    # If you don't have Hessian, uses L-BFGS.
    results = optimize(f, g, [127.0, 921.0])
    @assert norm(results.minimum - [0.0, 0.0]) < 0.01

    # If you have Hessian, uses Newton's method.
    results = optimize(f, g, h, [127.0, 921.0])
    @assert norm(results.minimum - [0.0, 0.0]) < 0.01

Note that `optimize()` has some simple rules you must follow to use it effectively:

* The function to be optimized must take in Float64 vectors as input.
* The raw function `f` must return a scalar.
* The gradient `g` must return a vector.
* The Hessian `h` must return a matrix.

## State of the Library

### Existing Functions
* Constant Step-Size Gradient Descent: `naive_gradient_descent()`
* Back-Tracking Line Search Gradient Descent: `gradient_descent()`
* Guarded Newton's Method: `newton()`
* Nelder-Mead Method: `nelder_mead()`
* Simulated Annealing: `simulated_annealing()`
* BFGS: `bfgs()`
* L-BFGS: `l_bfgs()`

### Planned Functions
* Brent's method
* Linear conjugate gradients
* Nonlinear conjugate gradients
* L-BFGS-B

### Wrapping Functions
* Will provide methods for wrapping functions to insure they satisfy usage rules.
* Will convert automatic conversion tools for input.
