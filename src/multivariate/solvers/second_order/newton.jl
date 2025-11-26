struct Newton{IL,L} <: SecondOrderOptimizer
    alphaguess!::IL
    linesearch!::L
end

"""
# Newton
## Constructor
```julia
Newton(; alphaguess = LineSearches.InitialStatic(),
linesearch = LineSearches.HagerZhang())
```

## Description
The `Newton` method implements Newton's method for optimizing a function. We use
a special factorization from the package `PositiveFactorizations.jl` to ensure
that each search direction is a direction of descent. See Wright and Nocedal and
Wright (ch. 6, 1999) for a discussion of Newton's method in practice.

## References
 - Nocedal, J. and S. J. Wright (1999), Numerical optimization. Springer Science 35.67-68: 7.
"""
function Newton(;
    alphaguess = LineSearches.InitialStatic(), # Good default for Newton
    linesearch = LineSearches.HagerZhang(),
)    # Good default for Newton
    Newton(_alphaguess(alphaguess), linesearch)
end

Base.summary(io::IO, ::Newton) = print(io, "Newton's Method")

mutable struct NewtonState{Tx,Tg,TH,T,F<:Cholesky} <: AbstractOptimizerState
    x::Tx
    g_x::Tg
    H_x::TH
    f_x::T
    x_previous::Tx
    f_x_previous::T
    F::F
    s::Tx
    @add_linesearch_fields()
end

function initial_state(method::Newton, options, d, initial_x)
    # TODO: Switch to `value_gradient_hessian!`
    f_x, g_x, H_x = NLSolversBase.value_gradient_hessian!!(d, initial_x)

    NewtonState(
        copy(initial_x), # Maintain current state in state.x
        copy(g_x), # Maintain current gradient in state.g_x
        copy(H_x), # Maintain current Hessian in state.H_x
        f_x, # Maintain current f in state.f_x
        fill!(similar(initial_x), NaN), # Maintain previous state in state.x_previous
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        Cholesky(similar(H_x, 0, 0), :U, 0),
        fill!(similar(initial_x), NaN), # Maintain current search direction in state.s
        @initial_linesearch()...,
    )
end

function update_state!(d, state::NewtonState, method::Newton)
    # Search direction is always the negative gradient divided by
    # a matrix encoding the absolute values of the curvatures
    # represented by H. It deviates from the usual "add a scaled
    # identity matrix" version of the modified Newton method. More
    # information can be found in the discussion at issue #153.

    if state.H_x isa AbstractSparseMatrix
        state.s .= .-(state.H_x \ convert(Vector, state.g_x))
    else
        state.F = cholesky!(Positive, state.H_x)
        if state.g_x isa Array
            # is this actually StridedArray?
            ldiv!(state.s, state.F, state.g_x)
            state.s .= .-state.s
        else
            # not Array, we can't do inplace ldiv
            gv = Vector{eltype(state.g_x)}(undef, length(state.g_x))
            gv .= .-state.g_x
            copyto!(state.s, state.F \ gv)
        end
    end
    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Update current position # x = x + alpha * s
    @. state.x = state.x + state.alpha * state.s
    return !lssuccess # break on linesearch error
end

function trace!(tr, d, state::NewtonState, iteration::Integer, ::Newton, options::Options, curr_time = time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g_x)
        dt["h(x)"] = copy(state.H_x)
        dt["Current step size"] = state.alpha
    end
    update!(
        tr,
        iteration,
        state.f_x,
        g_residual(state),
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
        options.callback,
    )
end
