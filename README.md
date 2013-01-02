# Optimization Functions for Julia

## Usage Examples

### Simple Function Demo

If you're just getting started, you probably want to use `optimize()`, which wraps the specific algorithms currently implemented and selects a good one based on the amount of information you can provide. See the examples below:

    load("Optim")
    using Optim

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

### Rosenbrock Function Demo

    load("src/Optim.jl")
    using Optim

    function rosenbrock(x::Vector)
      (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    end

    function rosenbrock_gradient(x::Vector)
      [-2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1],
       200.0 * (x[2] - x[1]^2)]
    end

    function rosenbrock_hessian(x::Vector)
      h = zeros(2, 2)
      h[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
      h[1, 2] = -400.0 * x[1]
      h[2, 1] = -400.0 * x[1]
      h[2, 2] = 200.0
      h
    end

    problem = Dict()
    problem[:f] = rosenbrock
    problem[:g] = rosenbrock_gradient
    problem[:h] = rosenbrock_hessian
    problem[:initial_x] = [0.0, 0.0]
    problem[:solution] = [1.0, 1.0]

    algorithms = ["naive_gradient_descent",
                  "gradient_descent",
                  "newton",
                  "bfgs",
                  "l-bfgs",
                  "nelder-mead",
                  "sa"]

    for algorithm = algorithms
      results = optimize(problem[:f],
                         problem[:g],
                         problem[:h],
                         problem[:initial_x],
                         algorithm,
                         10e-8,
                         true)
      print(results)
    end

### Curve Fitting Demo

There are also top-level methods `curve_fit()` and `estimate_errors()` that are useful for fitting data to non-linear models. See the following example:

    # a two-parameter exponential model
    model(xpts, p) = p[1]*exp(-xpts.*p[2])
    
    # some example data
    xpts = linspace(0,10,20)
    data = model(xpts, [1.0 2.0]) + 0.01*randn(length(xpts))
    
    beta, r, J = curve_fit(model, xpts, data, [0.5, 0.5])
	# beta = best fit parameters
	# r = vector of residuals
	# J = estimated Jacobian at solution
    
    # We can use these values to estimate errors on the fit parameters. To get 95% confidence error bars:
    errors = estimate_errors(beta, r, J)

## State of the Library

### Existing Functions
* Constant Step-Size Gradient Descent: `naive_gradient_descent()`
* Back-Tracking Line Search Gradient Descent: `gradient_descent()`
* Guarded Newton's Method: `newton()`
* BFGS: `bfgs()`
* L-BFGS: `l_bfgs()`
* Nelder-Mead Method: `nelder_mead()`
* Simulated Annealing: `simulated_annealing()`
* Levenberg-Marquardt: `levenberg_marquardt()`

### Planned Functions
* Brent's method
* Linear conjugate gradients
* Nonlinear conjugate gradients
* L-BFGS-B

### Wrapping Functions
* Will provide methods for wrapping functions to insure they satisfy usage rules
* Will convert automatic conversion tools for input
* Will provide automatic differentiation
* Will provide finite differencing
