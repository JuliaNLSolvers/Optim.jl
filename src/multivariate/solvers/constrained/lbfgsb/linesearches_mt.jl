# LineSearcher wrapper for the Moré-Thuente line search.
#
# Provides MTLS <: LineSearcher with a find_steplength method matching
# the HZAW interface so it can be used as a drop-in replacement in the
# L-BFGS-B optimizer.

include("mt/morethuente.jl")

"""
    MTLS

Moré-Thuente line search parameters, matching the `LineSearcher` interface.

    MTLS(; decrease=1e-4, curvature=0.9, maxiter=100)

Finds α > 0 satisfying the strong Wolfe conditions:

    ϕ(α)    ≤ ϕ(0) + μ·α·ϕ'(0)     (sufficient decrease)
    |ϕ'(α)| ≤ η·|ϕ'(0)|             (curvature)

where `μ = decrease` and `η = curvature`.

Based on: Moré, J. J., & Thuente, D. J. (1994). Line search algorithms
with guaranteed sufficient decrease. ACM TOMS, 20(3), 286–307.
"""
struct MTLS{T} <: LineSearcher
    μ::T          # sufficient decrease (Armijo)
    η::T          # curvature
    maxiter::Int
end

function MTLS(; decrease = 1e-4, curvature = 0.9, maxiter = 100)
    MTLS(decrease, curvature, maxiter)
end

MTLS{T}(mt::MTLS) where {T} = MTLS(T(mt.μ), T(mt.η), mt.maxiter)

Base.summary(::MTLS) = "Moré-Thuente Line Search"

"""
    find_steplength(mt::MTLS, φdφ, φ0, dφ0, c; αmax=Inf) -> (α, f, wolfe)

Adapter matching the `find_steplength(::HZAW, ...)` interface.

`φdφ(α)` returns `(f(α), f'(α))`.  `c` is the initial trial step.
`αmax` is passed through to the core algorithm which handles step
clamping internally (MINPACK style), ensuring that bracket state
always reflects the true evaluated positions.
"""
function find_steplength(mt::MTLS, φdφ, φ0, dφ0, c::T; αmax::T = T(Inf)) where {T}
    mt = MTLS{T}(mt)

    # Cap initial step at αmax
    c = min(c, αmax)

    try
        α, fα, dφα = more_thuente(
            φdφ,
            c,
            φ0,
            dφ0;
            μ = mt.μ,
            η = mt.η,
            max_iters = mt.maxiter,
            αmax = αmax,
        )

        # Verify strong Wolfe using derivative already computed by more_thuente
        wolfe = fα ≤ φ0 + mt.μ * α * dφ0 && abs(dφα) ≤ mt.η * abs(dφ0)
        return α, fα, wolfe
    catch e
        if e isa ErrorException && contains(e.msg, "did not converge")
            # Convergence failure → return NaN like HZAW does
            return T(NaN), T(NaN), false
        end
        rethrow()
    end
end
