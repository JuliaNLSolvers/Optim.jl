## Callbacks

Callbacks are functions that are called at certain points during the optimization process. They can be used to monitor progress, log information, or implement custom stopping criteria. Callbacks are called each **iteration** of an algorithm. By iteration, we mean each time the algorithm updates its current estimate of the solution and checks for convergence. This structure is not necessarily uniquely defined for all algorithms. For example, we could in principle call the callback function within the line search algorithm, or for each sampled point in a derivative-free algorithm.

### Callback Function Example

We show a simple example of a callback function that prints the current iteration number and objective value at each iteration.

```julia
using Optim
function my_callback(state)
    print(" Objective Value: ", state.f_x)
    println(" at state x: ", state.x)
    return false  # Return true to stop the optimization
end
function objective(x)
    return (x[1]-2)^2 + (x[2]-3)^2
end

initial_x = [0.0, 0.0]
method = BFGS()
options = Optim.Options(callback=my_callback)
d = OnceDifferentiable(objective, initial_x)

optstate = initial_state(method, options, d, initial_x)
result = optimize(d, initial_x, method, options, optstate)
```