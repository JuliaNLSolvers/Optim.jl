struct Newton{IL, L} <: SecondOrderOptimizer
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
function Newton(; alphaguess = LineSearches.InitialStatic(), # Good default for Newton
                linesearch = LineSearches.HagerZhang())    # Good default for Newton
    Newton(alphaguess, linesearch)
end

Base.summary(::Newton) = "Newton's Method"

mutable struct NewtonState{T, N, F<:Base.LinAlg.Cholesky} <: AbstractOptimizerState
    x::Array{T,N}
    x_previous::Array{T, N}
    f_x_previous::T
    F::F
    s::Array{T, N}
    @add_linesearch_fields()
end

function initial_state(method::Newton, options, d, initial_x::Array{T}) where T
    n = length(initial_x)
    # Maintain current gradient in gr
    s = similar(initial_x)

    value_gradient!!(d, initial_x)
    hessian!!(d, initial_x)
    
    NewtonState(copy(initial_x), # Maintain current state in state.x
                similar(initial_x), # Maintain previous state in state.x_previous
                T(NaN), # Store previous f in state.f_x_previous
                @static(VERSION >= v"0.7.0-DEV.393" ?
                        Base.LinAlg.Cholesky(similar(d.H, T, 0, 0), :U, BLAS.BlasInt(0)) :
                        Base.LinAlg.Cholesky(similar(d.H, T, 0, 0), :U)),
                similar(initial_x), # Maintain current search direction in state.s
                @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!(d, state::NewtonState{T}, method::Newton) where T
    # Search direction is always the negative gradient divided by
    # a matrix encoding the absolute values of the curvatures
    # represented by H. It deviates from the usual "add a scaled
    # identity matrix" version of the modified Newton method. More
    # information can be found in the discussion at issue #153.
    update_h!(d, state, method)
    if typeof(NLSolversBase.hessian(d)) <: AbstractSparseMatrix
        state.s .= -NLSolversBase.hessian(d)\gradient(d)
    else
        state.F = cholfact!(Positive, NLSolversBase.hessian(d))
        state.s .= -(state.F\gradient(d))
    end
    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Update current position # x = x + alpha * s
    @. state.x = state.x + state.alpha * state.s
    lssuccess == false # break on linesearch error
end

function assess_convergence(state::NewtonState, d, options)
  default_convergence_assessment(state, d, options)
end

function trace!(tr, d, state, iteration, method::Newton, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(gradient(d))
        dt["h(x)"] = copy(NLSolversBase.hessian(d))
        dt["Current step size"] = state.alpha
    end
    g_norm = vecnorm(gradient(d), Inf)
    update!(tr,
            iteration,
            value(d),
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end
