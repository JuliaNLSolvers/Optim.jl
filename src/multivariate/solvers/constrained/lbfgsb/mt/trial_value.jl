# Trial value selection for the More-Thuente line search.
#
# Implements the four-case analysis of Fortran dcstep (MINPACK-2), which
# runs regardless of bracket status. Local step bounds (stpmin, stpmax)
# computed by the caller guide the extrapolation when not yet bracketed.
# TODO maybe expose?
const BRACKET_SAFEGUARD = 66 // 100
"""
    trial_value(αₗ, fₗ, gₗ, αᵤ, fᵤ, gᵤ, αₜ, fₜ, gₜ, bracketed,
                stpmin, stpmax) -> αₜ⁺

Compute the next trial value for the line search.

Implements the four-case analysis of Fortran `dcstep` (MINPACK-2).
Unlike the paper-faithful version that only extrapolates when not
bracketed, this runs the full case analysis in both states.

`stpmin` and `stpmax` are local step bounds from the caller:
- **Bracketed**: `min(αₗ,αᵤ)` / `max(αₗ,αᵤ)` — bracket interval.
- **Not bracketed**: `αₜ + 1.1·(αₜ - αₗ)` / `αₜ + 4·(αₜ - αₗ)` —
  extrapolation window (Fortran `xtrapl`/`xtrapu`).

# Cases (Fortran dcstep / paper §4)

- **Case 1** (`fₜ > fₗ`): trial rose. Cubic+quadratic between αₗ, αₜ.
  Bracket-independent.
- **Case 2** (opposite slopes): cubic+secant, farther from αₜ.
  Bracket-independent.
- **Case 3** (same slopes, `|gₜ| ≤ |gₗ|`):
  - *Bracketed*: closer of cubic/secant + 0.66 safeguard.
  - *Not bracketed*: farther of cubic/secant, clamped to [stpmin,stpmax].
- **Case 4** (same slopes, `|gₜ| > |gₗ|`):
  - *Bracketed*: cubic between αₜ and αᵤ.
  - *Not bracketed*: jump to stpmax (or stpmin).
"""
function trial_value(αₗ, fₗ, gₗ, αᵤ, fᵤ, gᵤ, αₜ, fₜ, gₜ, bracketed, stpmin, stpmax)
    if fₜ > fₗ
        # Case 1: trial rose — minimum bracketed in (αₗ, αₜ).
        # Cubic α_c and quadratic α_q. Take α_c if closer to αₗ,
        # else average — avoids pathological α_q near αₗ when fₜ ≫ fₗ.
        α_c = cubic_min(αₗ, fₗ, gₗ, αₜ, fₜ, gₜ)
        α_q = quadratic_min(αₗ, fₗ, gₗ, αₜ, fₜ)
        return abs(α_c - αₗ) < abs(α_q - αₗ) ? α_c : (α_c + α_q) / 2

    elseif gₜ * gₗ < zero(gₜ)
        # Case 2: lower value, opposite slope signs — minimum bracketed
        # by slope sign change. Take whichever of cubic/secant is farther
        # from αₜ.
        α_c = cubic_min(αₗ, fₗ, gₗ, αₜ, fₜ, gₜ)
        α_s = secant_min(αₗ, gₗ, αₜ, gₜ)
        return abs(α_c - αₜ) ≥ abs(α_s - αₜ) ? α_c : α_s

    elseif abs(gₜ) ≤ abs(gₗ)
        # Case 3: lower value, same slope signs, magnitude decreasing.
        α_c = cubic_min(αₗ, fₗ, gₗ, αₜ, fₜ, gₜ)
        α_s = secant_min(αₗ, gₗ, αₜ, gₜ)

        # Cubic is usable only if it exists (finite) and its minimiser
        # lies past αₜ away from αₗ. Matches Fortran: r < 0 && γ ≠ 0.
        cubic_ok = isfinite(α_c) && sign(αₜ - αₗ) * (α_c - αₜ) > zero(α_c)

        if bracketed
            # Closer of cubic/secant, with 0.66 safeguard toward αᵤ.
            αₜ⁺ = if cubic_ok
                abs(α_c - αₜ) < abs(α_s - αₜ) ? α_c : α_s
            else
                α_s
            end
            if αₜ > αₗ
                return min(αₜ + BRACKET_SAFEGUARD * (αᵤ - αₜ), αₜ⁺)
            else
                return max(αₜ + BRACKET_SAFEGUARD * (αᵤ - αₜ), αₜ⁺)
            end
        else
            # Not bracketed: farther of cubic/secant, clamped to local
            # bounds. When cubic unusable, fall back to stpmax/stpmin
            # (Fortran dcstep lines 3865-3869).
            α_c_eff = cubic_ok ? α_c : (αₜ > αₗ ? stpmax : stpmin)
            αₜ⁺ = abs(α_c_eff - αₜ) > abs(α_s - αₜ) ? α_c_eff : α_s
            return clamp(αₜ⁺, stpmin, stpmax)
        end

    else
        # Case 4: lower value, same slope signs, magnitude increasing.
        if bracketed
            # Cubic between αₜ and αᵤ.
            return cubic_min(αₜ, fₜ, gₜ, αᵤ, fᵤ, gᵤ)
        else
            # Jump to boundary (Fortran dcstep lines 3919-3923).
            return αₜ > αₗ ? stpmax : stpmin
        end
    end
end
