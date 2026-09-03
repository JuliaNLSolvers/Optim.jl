# Polynomial minimizers used by the More-Thuente line search (§4 of
# Moré & Thuente, 1994, ACM TOMS 20(3):286-307).
#
# Three closed-form interpolating-polynomial minimizers, matching the
# paper's notation:
#
#   α_c: cubic Hermite minimizer   (4 data: fₗ, gₗ, fₜ, gₜ)
#   α_q: quadratic minimizer       (3 data: fₗ, gₗ, fₜ)
#   α_s: secant on the derivative  (3 data: fₗ, gₗ, gₜ)
#
# Each function returns the α at which the corresponding interpolant has
# a stationary point (a minimum in the cases the paper actually uses
# them). They are agnostic to whether the input values come from φ or
# from the auxiliary function ψ.

"""
    cubic_min(αₗ, fₗ, gₗ, αₜ, fₜ, gₜ) -> α_c

Minimizer of the unique cubic Hermite polynomial p(α) satisfying
    p(αₗ) = fₗ,  p'(αₗ) = gₗ,
    p(αₜ) = fₜ,  p'(αₜ) = gₜ.

Returns `NaN` of the appropriate type when the cubic has no real
stationary point. This happens when the discriminant θ² - gₗ·gₜ is
negative — possible only when gₗ and gₜ have the same sign (paper's
Case 3); in that situation the data is consistent with the cubic being
monotone over the relevant region.

The function is symmetric in its two interpolation points: swapping
(αₗ, fₗ, gₗ) with (αₜ, fₜ, gₜ) yields the same α_c. The sign convention
on γ below picks the minimizer (where p'' ≥ 0), not the maximizer.
"""
function cubic_min(αₗ, fₗ, gₗ, αₜ, fₜ, gₜ)
    # θ is the standard "modified second difference" of cubic Hermite
    # interpolation (Nocedal & Wright eq. 3.59, d₁):
    #     θ = gₗ + gₜ + 3·(fₗ - fₜ)/(αₜ - αₗ)
    θ = 3*(fₗ - fₜ)/(αₜ - αₗ) + gₗ + gₜ

    # The minimizer formula involves √(θ² - gₗ·gₜ). Rescale by
    # s = max(|θ|, |gₗ|, |gₜ|) before squaring so that the squared
    # quantities do not over/underflow when slopes are very large or
    # very small (this is MINPACK's `s` trick).
    s = max(abs(θ), abs(gₗ), abs(gₜ))
    s == zero(s) && return (αₗ + αₜ)/2          # degenerate, fall back to midpoint
    disc = (θ/s)^2 - (gₗ/s)*(gₜ/s)

    # No real stationary point.
    disc < zero(disc) && return oftype(αₗ, NaN)

    # γ ≡ sign(αₜ - αₗ) · √(θ² - gₗ·gₜ). The sign selects the
    # minimum-side root of p'(α) = 0 rather than the maximum-side root.
    γ = s * sqrt(disc) * sign(αₜ - αₗ)

    # Closed form (Nocedal & Wright 3.59):
    #     α_c = αₜ - (αₜ - αₗ)·(gₜ + γ - θ)/(gₜ - gₗ + 2γ)
    return αₜ - (αₜ - αₗ) * (gₜ + γ - θ) / (gₜ - gₗ + 2γ)
end


"""
    quadratic_min(αₗ, fₗ, gₗ, αₜ, fₜ) -> α_q

Minimizer of the unique quadratic q(α) interpolating
    q(αₗ) = fₗ,  q'(αₗ) = gₗ,  q(αₜ) = fₜ.

This is the paper's α_q (Case 1 of §4), used when fₜ > fₗ and we want
a candidate biased toward the lower-valued endpoint αₗ.

Note the asymmetry: the slope at αₗ is used; the slope at αₜ is not.
"""
function quadratic_min(αₗ, fₗ, gₗ, αₜ, fₜ)
    # q(α) = fₗ + gₗ(α - αₗ) + a(α - αₗ)²
    # q(αₜ) = fₜ pins  a = (fₜ - fₗ - gₗ·h)/h²,  with h = αₜ - αₗ.
    # q'(α_q) = 0      ⇒  α_q = αₗ - gₗ/(2a)
    #                  ⇒  α_q = αₗ + h · gₗ / (2·(gₗ - (fₜ - fₗ)/h)).
    secant_slope = (fₜ - fₗ) / (αₜ - αₗ)
    return αₗ + (αₜ - αₗ) * gₗ / (2 * (gₗ - secant_slope))
end


"""
    secant_min(αₗ, gₗ, αₜ, gₜ) -> α_s

Zero of the linear interpolant of gₗ at αₗ and gₜ at αₜ — equivalently,
the minimizer of any quadratic whose derivative is that linear
interpolant.

This is the paper's α_s (Cases 2 and 3 of §4). The result depends only
on the slopes; function values do not enter the formula.

Symmetric in its two interpolation points.
"""
function secant_min(αₗ, gₗ, αₜ, gₜ)
    # Linear interpolant of g over [αₗ, αₜ] is zero at
    #     α_s = αₜ + (αₗ - αₜ)·gₜ/(gₜ - gₗ).
    return αₜ + (αₗ - αₜ) * gₜ / (gₜ - gₗ)
end
