# Optimization Functions for Julia

## Existing Functions
* Constant Step-Size Gradient Descent: `naive_gradient_descent()`
* Back-Tracking Line Search Gradient Descent: `gradient_descent()`
* Guarded Newton's Method: `newton()`
* Nelder-Mead Method: `nelder_mead()`
* Simulated Annealing: `simulated_annealing()`
* BFGS: `bfgs()`
* L-BFGS: `l_bfgs()`

## Planned Functions
* Linear conjugate gradients
* Nonlinear conjugate gradients
* L-BFGS-B

## Rules of Usage
* Functions to be optimized must take in Float64 vectors as input.
* The raw function `f` returns a scalar.
* The gradient `g` returns a vector.
* The Hessian `h` returns a matrix.

## Wrapping Functions
* Will provide methods for wrapping functions to insure they satisfy usage rules.
* Will convert automatic conversion tools for input.
