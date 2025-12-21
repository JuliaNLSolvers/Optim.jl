## Optimization State

Each algorithm in Optim.jl maintains an optimization state that encapsulates all relevant information about the current iteration of the optimization process. This state is represented by the sub-types of `Optim.OptimizationState` and contains various fields that provide insights into the progress of the optimization and any information needed to maintain and update the search direction.

### Exceptions

Currently, there are two main exceptions to this structure:
- **SAMIN**: This algorithm is currently not written using the main `optimize` loop and does not maintain an `OptimizationState`.
- **Univariate Optimization Algorithms**: These algorithms do not use the `OptimizationState` structure as they also do not use the main `optimize` loop.

The exceptions matter mostly for users who want to pre-allocate the `OptimizationState` for performance reasons. In these cases, users should check the documentation of the specific algorithm they are using to see if it supports pre-allocation. It also matters for users who want to make use of the callback functionality, as the callback functions receive the `OptimizationState` as an argument. If the algorithm does not use the `OptimizationState`, the callback will instead receive a `NamedTuple` with relevant information and the callback functions should not use type annotations for their arguments based on the `OptimizationState` hierarchy.

### Using the Optimization State

As mentioned above, the optimization state is passed to callback functions during the optimization process. Users can access various fields of the state to monitor progress or implement custom logic based on the current state of the optimization. It is also possible to pre-allocate the optimization state if users which to re-use it across multiple optimization runs for performance reasons. This can be done using the `initial_state` function, which takes the optimization method, options, differentiable object, and initial parameters as arguments.

#### Initial State Example
```julia
using Optim
function objective(x)
    return (x[1]-2)^2 + (x[2]-3)^2
end

initial_x = [0.0, 0.0]
method = BFGS()
options = Optim.Options(callback=my_callback)
d = OnceDifferentiable(objective, initial_x)

# Pre-allocate the optimization state
optstate = initial_state(method, options, d, initial_x)

# Verify that the state has the properties f_x and x
hasproperty(optstate, :f_x)  # true
hasproperty(optstate, :x)    # true

result = optimize(d, initial_x, method, options, optstate)
```

After the optimization is complete, the state has been updated as part of the optimization process and contains information about the final iteration. Users can access fields of the state to retrieve information about the final state. For example, we can verify that the final objective value matches the value stored in the state.

```julia
@assert optstate.f_x == Optim.minimum(result)
```