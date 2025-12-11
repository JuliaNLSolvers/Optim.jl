#
# Conjugate gradient
#
# This is an independent implementation of:
#   W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a
#     conjugate gradient method with guaranteed descent. ACM
#     Transactions on Mathematical Software 32: 113–137.
#
# Code comments such as "HZ, stage X" or "HZ, eqs Y" are with
# reference to a particular point in this paper.
#
# Several aspects of the following have also been incorporated:
#   W. W. Hager and H. Zhang (2013) The limited memory conjugate
#     gradient method.
#
# This paper will be denoted HZ2013 below.
#
#
# There are some modifications and/or extensions from what's in the
# paper (these may or may not be extensions of the cg_descent code
# that can be downloaded from Hager's site; his code has undergone
# numerous revisions since publication of the paper):
#
# cgdescent: the termination condition employs a "unit-correct"
#   expression rather than a condition on gradient
#   components---whether this is a good or bad idea will require
#   additional experience, but preliminary evidence seems to suggest
#   that it makes "reasonable" choices over a wider range of problem
#   types.
#
# both: checks for Inf/NaN function values
#
# both: support maximum value of alpha (equivalently, c). This
#   facilitates using these routines for constrained minimization
#   when you can calculate the distance along the path to the
#   disallowed region. (When you can't easily calculate that
#   distance, it can still be handled by returning Inf/NaN for
#   exterior points. It's just more efficient if you know the
#   maximum, because you don't have to test values that won't
#   work.) The maximum should be specified as the largest value for
#   which a finite value will be returned.  See, e.g., limits_box
#   below.  The default value for alphamax is Inf. See alphamaxfunc
#   for cgdescent and alphamax for linesearch_hz.

struct ConjugateGradient{Tf,T,Tprep,IL,L} <: FirstOrderOptimizer
    eta::Tf
    P::T
    precondprep!::Tprep
    alphaguess!::IL
    linesearch!::L
    manifold::Manifold
end

Base.summary(io::IO, ::ConjugateGradient) = print(io, "Conjugate Gradient")

"""
# Conjugate Gradient Descent
## Constructor
```julia
ConjugateGradient(; alphaguess = LineSearches.InitialHagerZhang(),
linesearch = LineSearches.HagerZhang(),
eta = 0.4,
P = nothing,
precondprep = Returns(nothing),
manifold = Flat())
```
The strictly positive constant ``eta`` is used in determining
the next step direction, and the default here deviates from the one used in the
original paper (where it was ``0.01``). See more details in the original papers
referenced below.

## Description
The `ConjugateGradient` method implements Hager and Zhang (2006) and elements
from Hager and Zhang (2013). Notice, the default `linesearch` is `HagerZhang`
from LineSearches.jl. This line search is exactly the one proposed in Hager and
Zhang (2006).

## References
 - W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent. ACM Transactions on Mathematical Software 32: 113-137.
 - W. W. Hager and H. Zhang (2013), The Limited Memory Conjugate Gradient Method. SIAM Journal on Optimization, 23, pp. 2150-2168.
"""
function ConjugateGradient(;
    alphaguess = LineSearches.InitialHagerZhang(),
    linesearch = LineSearches.HagerZhang(),
    eta::Real = 0.4,
    P::Any = nothing,
    precondprep = Returns(nothing),
    manifold::Manifold = Flat(),
)

    ConjugateGradient(eta, P, precondprep, _alphaguess(alphaguess), linesearch, manifold)
end

mutable struct ConjugateGradientState{Tx,T,G} <: AbstractOptimizerState
    x::Tx
    g_x::G
    f_x::T
    x_previous::Tx
    g_x_previous::G
    f_x_previous::T
    y::Tx
    py::Tx
    pg::Tx
    s::Tx
    @add_linesearch_fields()
end

function reset!(cg::ConjugateGradient, cgs::ConjugateGradientState, obj, x)
    copyto!(cgs.x, x)
    retract!(cg.manifold, cgs.x)
    f_x, g_x = NLSolversBase.value_gradient!(obj, cgs.x)
    copyto!(cgs.g_x, g_x)
    project_tangent!(cg.manifold, cgs.g_x, cgs.x)
    cgs.f_x = f_x

    fill!(cgs.x_previous, NaN)
    cgs.f_x_previous = oftype(cgs.f_x_previous, NaN)
    fill!(cgs.g_x_previous, NaN)

    _precondition!(cgs.pg, cg, cgs.x, cgs.g_x)
    if cg.P !== nothing
        project_tangent!(cg.manifold, cgs.pg, cgs.x)
    end
    cgs.s .= .-cgs.pg

    return nothing
end
function initial_state(method::ConjugateGradient, ::Options, d, initial_x)
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)
    f_x, g_x = value_gradient!(d, initial_x)
    g_x = copy(g_x)
    project_tangent!(method.manifold, g_x, initial_x)

    # Could move this out? as a general check?
    #=
    # Output messages
    isfinite(value(d)) || error("Initial f(x) is not finite ($(value(d)))")
    if !all(isfinite, gradient(d))
        @show gradient(d)
        @show find(.!isfinite.(gradient(d)))
        error("Gradient must have all finite values at starting point")
    end
    =#
    # Determine the intial search direction
    #    if we don't precondition, then this is an extra superfluous copy
    #    TODO: consider allowing a reference for pg instead of a copy
    pg = copy(g_x)
    _precondition!(pg, method, initial_x, g_x)
    if method.P !== nothing
        project_tangent!(method.manifold, pg, initial_x)
    end

    ConjugateGradientState(
        initial_x, # Maintain current state in state.x
        g_x, # Maintain current gradient in state.g
        f_x, # Maintain current f in state.f
        fill!(similar(initial_x), NaN), # Maintain previous state in state.x_previous
        fill!(similar(g_x), NaN), # Store previous gradient in state.g_x_previous
        oftype(f_x, NaN), # Store previous f in state.f_x_previous
        0 .* (initial_x), # Intermediate value in CG calculation
        0 .* (initial_x), # Preconditioned intermediate value in CG calculation
        pg, # Maintain the preconditioned gradient in pg
        -pg, # Maintain current search direction in state.s
        @initial_linesearch()...,
    )
end

function update_state!(d, state::ConjugateGradientState, method::ConjugateGradient)
    # Search direction is predetermined

    # Maintain a record of the previous gradient
    copyto!(state.g_x_previous, state.g_x)

    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Update current position # x = x + alpha * s
    state.x .= muladd.(state.alpha, state.s, state.x)
    retract!(method.manifold, state.x)

    # Update the function value and gradient
    f_x, g_x = NLSolversBase.value_gradient!(d, state.x)
    copyto!(state.g_x, g_x)
    project_tangent!(method.manifold, state.g_x, state.x)
    state.f_x = f_x

    # Check sanity of function and gradient
    isfinite(f_x) || error(LazyString("Non-finite f(x) while optimizing (", f_x, ")"))

    # Determine the next search direction using HZ's CG rule
    #  Calculate the beta factor (HZ2013)
    # -----------------
    # Comment on py: one could replace the computation of py with
    #    ydotpgprev = dot(y, pg)
    #    dot(y, py)  >>>  dot(y, pg) - ydotpgprev
    # but I am worried about round-off here, so instead we make an
    # extra copy, which is probably minimal overhead.
    # -----------------
    # also updates P for the preconditioning step below
    _apply_precondprep(method, state.x)
    dPd = _inverse_precondition(method, state)
    etak = method.eta * real(dot(state.s, state.g_x_previous)) / dPd # New in HZ2013
    state.y .= state.g_x .- state.g_x_previous
    ydots = real(dot(state.y, state.s))
    copyto!(state.py, state.pg)        # below, store pg - pg_previous in py
    # P already updated in _apply_precondprep above
    __precondition!(state.pg, method.P, g_x)

    state.py .= state.pg .- state.py
    # ydots may be zero if f is not strongly convex or the line search does not satisfy Wolfe
    betak =
        (
            real(dot(state.y, state.pg)) -
            real(dot(state.y, state.py)) * real(dot(state.g_x, state.s)) / ydots
        ) / ydots
    # betak may be undefined if ydots is zero (may due to f not strongly convex or non-Wolfe linesearch)
    beta = NaNMath.max(betak, etak) # TODO: Set to zero if betak is NaN?
    state.s .= beta .* state.s .- state.pg
    project_tangent!(method.manifold, state.s, state.x)
    return !lssuccess # break on linesearch error
end

# Function value, gradient and Hessian are already updated in `update_state!`
update_fgh!(d, state, ::ConjugateGradient) = nothing

function trace!(
    tr,
    d,
    state,
    iteration,
    method::ConjugateGradient,
    options,
    curr_time = time(),
)
    common_trace!(tr, d, state, iteration, method, options, curr_time)
end
