# Main loop for the More-Thuente line search.
#
# Implementation of Algorithm 2.6 (Moré & Thuente, 1994) with MINPACK-2
# engineering from the Fortran dcsrch/dcstep bundled with L-BFGS-B 3.0:
# local step bounds (stmin/stmax), Fortran-style stage transition,
# conditional ψ-usage, boundary termination, and fallback to best point.

include("poly_min.jl")    # cubic_min, quadratic_min, secant_min
include("update.jl")      # update_bracket   (Section 2 / Section 3 case analysis)
include("trial_value.jl") # trial_value      (dcstep four-case analysis)

"""
    more_thuente(ϕdϕ, α0, ϕ0, dϕ0; μ=1e-4, η=0.9, max_iters=100,
                 αmax=Inf, xtol=0.1) -> (α, ϕ(α), ϕ'(α))

Find α > 0 satisfying the strong Wolfe conditions

    ϕ(α)    ≤ ϕ(0) + μ·α·ϕ'(0)     (sufficient decrease)
    |ϕ'(α)| ≤ η·|ϕ'(0)|             (curvature)

`ϕdϕ(α)` must return `(ϕ(α), ϕ'(α))`. `ϕ0 = ϕ(0)` and `dϕ0 = ϕ'(0)` are
passed in so the search doesn't need to evaluate at α = 0. `α0 > 0` is
the initial trial step.

When `αmax` is finite the search is constrained to `(0, αmax]`. Trial
values are clamped internally and the algorithm tracks local step bounds
`(stmin, stmax)` matching Fortran `dcsrch`. If the trial reaches `αmax`
with sufficient decrease and the function still descending, the boundary
point is accepted.

`xtol` (default 0.1, matching L-BFGS-B 3.0) controls the relative
bracket-width tolerance for the stall fallback.
"""
function more_thuente(ϕdϕ, α0, ϕ0, dϕ0; μ=1e-4, η=0.9, max_iters=100,
                      αmax=oftype(α0, Inf), xtol=oftype(α0, 1//10))
    @assert dϕ0 < 0 "Search direction is not a descent direction"

    μdϕ0 = μ * dϕ0

    # State: best endpoint αₗ, other endpoint αᵤ, current trial αₜ.
    # Values stored as ϕ; we convert to ψ at call sites when needed.
    αₗ, fₗ, gₗ = zero(α0), ϕ0, dϕ0
    αᵤ, fᵤ, gᵤ = zero(α0), ϕ0, dϕ0
    αₜ = α0

    bracketed = false
    stage1    = true

    # Bracket-shrink monitor (paper p. 293): if the bracket width
    # doesn't shrink by factor 0.66 over the past two trials, force
    # bisection.
    width      = oftype(α0, Inf)
    width_prev = oftype(α0, Inf)

    # Local step bounds (Fortran dcsrch lines 3537-3538).
    # Initial: stmin=0, stmax = α₀ + 4·α₀.
    stmin = zero(α0)
    stmax = α0 + oftype(α0, 4) * α0

    for _ in 1:max_iters
        fₜ, gₜ = ϕdϕ(αₜ)
        ftest = ϕ0 + αₜ * μdϕ0

        # ── Termination checks ──────────────────────────────────────

        # Strong Wolfe — convergence.
        if fₜ ≤ ftest && abs(gₜ) ≤ η * abs(dϕ0)
            return αₜ, fₜ, gₜ
        end

        # Boundary termination (MINPACK "STP = STPMAX"):
        # at αmax with sufficient decrease and still descending.
        if αₜ == αmax && fₜ ≤ ftest && gₜ ≤ μdϕ0
            return αₜ, fₜ, gₜ
        end

        # ── Stage transition (Fortran dcsrch line 3573) ─────────────
        # Fortran uses g ≥ 0 (the actual derivative is non-negative),
        # stricter than the paper's ψ'(αₜ) ≥ 0 condition.
        if stage1 && fₜ ≤ ftest && gₜ ≥ zero(gₜ)
            stage1 = false
        end

        # ── Trial value + bracket update ────────────────────────────
        # Fortran dcsrch only feeds ψ-values to dcstep when all three
        # hold: stage 1, function decreased from best (fₜ ≤ fₗ), and
        # sufficient decrease not yet met (fₜ > ftest). Otherwise φ
        # is used even in stage 1 (dcsrch line 3600).
        use_modified = stage1 && fₜ ≤ fₗ && fₜ > ftest

        αₜ_next = if use_modified
            fₗψ, gₗψ = fₗ - ϕ0 - αₗ*μdϕ0, gₗ - μdϕ0
            fᵤψ, gᵤψ = fᵤ - ϕ0 - αᵤ*μdϕ0, gᵤ - μdϕ0
            fₜψ, gₜψ = fₜ - ϕ0 - αₜ*μdϕ0, gₜ - μdϕ0

            αt⁺ = trial_value(αₗ, fₗψ, gₗψ, αᵤ, fᵤψ, gᵤψ,
                              αₜ, fₜψ, gₜψ, bracketed, stmin, stmax)
            (αₗ, fₗψ, gₗψ, αᵤ, fᵤψ, gᵤψ, bracketed) =
                update_bracket(αₗ, fₗψ, gₗψ, αᵤ, fᵤψ, gᵤψ,
                               αₜ, fₜψ, gₜψ, bracketed)

            # Convert returned ψ-values back to ϕ for storage.
            fₗ, gₗ = fₗψ + ϕ0 + αₗ*μdϕ0, gₗψ + μdϕ0
            fᵤ, gᵤ = fᵤψ + ϕ0 + αᵤ*μdϕ0, gᵤψ + μdϕ0
            αt⁺
        else
            αt⁺ = trial_value(αₗ, fₗ, gₗ, αᵤ, fᵤ, gᵤ,
                              αₜ, fₜ, gₜ, bracketed, stmin, stmax)
            (αₗ, fₗ, gₗ, αᵤ, fᵤ, gᵤ, bracketed) =
                update_bracket(αₗ, fₗ, gₗ, αᵤ, fᵤ, gᵤ,
                               αₜ, fₜ, gₜ, bracketed)
            αt⁺
        end

        # ── Bracket-shrink safeguard (paper p. 293) ────────────────
        if bracketed
            if abs(αᵤ - αₗ) ≥ BRACKET_SAFEGUARD * width_prev
                αₜ_next = (αₗ + αᵤ) / 2
            end
            width_prev = width
            width      = abs(αᵤ - αₗ)
        end

        # ── Update local step bounds for next iteration ─────────────
        # (Fortran dcsrch lines 3642-3648).
        if bracketed
            stmin = min(αₗ, αᵤ)
            stmax = max(αₗ, αᵤ)
        else
            stmin = αₜ_next + oftype(αₜ_next, 1.1) * (αₜ_next - αₗ)
            stmax = αₜ_next + oftype(αₜ_next, 4.0) * (αₜ_next - αₗ)
        end

        # ── Global step bounds (Fortran lines 3652-3653) ───────────
        αₜ_next = max(αₜ_next, zero(αₜ_next))
        αₜ_next = min(αₜ_next, αmax)

        # ── Fallback to best point when progress stalls ─────────────
        # (Fortran dcsrch lines 3578-3585, 3658-3659). Return the
        # best point found rather than looping — equivalent to the
        # Fortran's WARNING termination on the next call.
        if bracketed && (αₜ_next ≤ stmin || αₜ_next ≥ stmax ||
                         stmax - stmin ≤ xtol * stmax)
            return αₗ, fₗ, gₗ
        end

        αₜ = αₜ_next
    end

    error("more_thuente did not converge in $max_iters iterations")
end
