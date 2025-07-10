struct Newton{IL,L,S} <: SecondOrderOptimizer
    alphaguess!::IL
    linesearch!::L
    solve::S  # Function that takes (H, g) -> s where H*s = -g
end


"""
# Newton
## Constructor
```julia
Newton(; alphaguess = LineSearches.InitialStatic(),
       linesearch = LineSearches.HagerZhang(),
       solve = default_newton_solve)
```
## Description
The `Newton` method implements Newton's method for optimizing a function. 

The `solve` function should take (H, g) and return s such that H*s = -g.
Defaults to a robust solver that handles dense or sparse matrices.
To do this, we use a special factorization from the package `PositiveFactorizations.jl` to ensure
that each search direction is a direction of descent. See Wright and Nocedal and
Wright (ch. 6, 1999) for a discussion of Newton's method in practice.

## References
 - Nocedal, J. and S. J. Wright (1999), Numerical optimization. Springer Science 35.67-68: 7.
"""
function Newton(;
    alphaguess = LineSearches.InitialStatic(),
    linesearch = LineSearches.HagerZhang(), 
    solve = default_newton_solve
)
    Newton(_alphaguess(alphaguess), linesearch, solve)
end

# Default solver that handles common matrix types intelligently
function default_newton_solve(H, g)
    if H isa AbstractSparseMatrix
        return -(H \ g)
    else
        # Use PositiveFactorizations for robustness on dense matrices
        F = cholesky(Positive, H)
        return -(F \ g)
    end
end

Base.summary(::Newton) = "Newton's Method"

mutable struct NewtonState{Tx,T} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx  
    f_x_previous::T
    s::Tx
    @add_linesearch_fields()
end

function initial_state(method::Newton, options, d, initial_x)
    T = eltype(initial_x)
    
    value_gradient!!(d, initial_x)
    hessian!!(d, initial_x)
    
    NewtonState(
        copy(initial_x),     # Current state
        copy(initial_x),     # Previous state
        T(NaN),             # Previous function value  
        similar(initial_x), # Search direction
        @initial_linesearch()...,
    )
end

function update_state!(d, state::NewtonState, method::Newton)
    H = NLSolversBase.hessian(d)
    g = gradient(d)
    
    # Clean and simple - just call the user's solve function
    state.s .= method.solve(H, g)
    
    # Perform line search
    lssuccess = perform_linesearch!(state, method, d)
    
    # Update position
    @. state.x = state.x + state.alpha * state.s
    
    return !lssuccess
end

function trace!(tr, d, state, iteration, method::Newton, options, curr_time = time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(gradient(d))
        dt["h(x)"] = copy(NLSolversBase.hessian(d))
        dt["Current step size"] = state.alpha
    end
    g_norm = norm(gradient(d), Inf)
    update!(
        tr,
        iteration,
        value(d),
        g_norm,
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
        options.callback,
    )
end
