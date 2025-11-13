struct Newton{IL,L,S} <: SecondOrderOptimizer
    alphaguess!::IL
    linesearch!::L
    solve!::S   # mutating solver: (d, state, method) i.e., writes state.s (and maybe state.F)
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
If the matrix is not an `AbstractSparseMatrix`, we use a special factorization from the package `PositiveFactorizations.jl` to ensure
that each search direction is a direction of descent. See Wright and Nocedal and
Wright (ch. 6, 1999) for a discussion of Newton's method in practice.

## References
 - Nocedal, J. and S. J. Wright (1999), Numerical optimization. Springer Science 35.67-68: 7.
"""
function Newton(;
    alphaguess = LineSearches.InitialStatic(),
    linesearch = LineSearches.HagerZhang(),
    solve = default_newton_solve!
)
    Newton(_alphaguess(alphaguess), linesearch, solve)
end

mutable struct NewtonState{Tx,T,F<:Cholesky} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx  
    f_x_previous::T
    F::F
    s::Tx
    @add_linesearch_fields()
end

function initial_state(method::Newton, options, d, initial_x)
    T = eltype(initial_x)

    value_gradient!!(d, initial_x)
    hessian!!(d, initial_x)

    NewtonState(
        copy(initial_x),                  # x
        copy(initial_x),                  # x_previous
        T(NaN),                           # f_x_previous
        Cholesky(similar(d.H, T, 0, 0),   # F
                :U, 0),
        similar(initial_x),               # s
        @initial_linesearch()...,
    )
end

# Default solver that handles common matrix types intelligently
function default_newton_solve!(d, state::NewtonState, method::Newton)
    H = NLSolversBase.hessian(d)
    g = gradient(d)
    T = eltype(state.x)

    if H isa AbstractSparseMatrix
        state.s .= .-(H \ convert(Vector{T}, gradient(d)))
    else
        # Use PositiveFactorizations for robustness on dense matrices
         # Search direction is always the negative gradient divided by
         # a matrix encoding the absolute values of the curvatures
         # represented by H. It deviates from the usual "add a scaled
         # identity matrix" version of the modified Newton method. More
         # information can be found in the discussion at issue #153.
         state.F = cholesky!(Positive, H)
         if g isa StridedArray
            ldiv!(state.s, state.F, g)
            state.s .= .-state.s
         else
            gv = Vector{T}(undef, length(g))
            gv .= .-g
            copyto!(state.s, state.F \ gv)
         end
    end
end

Base.summary(io::IO, ::Newton) = print(io, "Newton's Method")

function update_state!(d, state::NewtonState, method::Newton)
    method.solve!(d, state, method)  # should mutate state.s (and maybe state.F)

    lssuccess = perform_linesearch!(state, method, d)
    @. state.x = state.x + state.alpha * state.s
    return !lssuccess
end

function trace!(tr, d, state::NewtonState, iteration::Integer, method::Newton, options::Options, curr_time = time())
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
