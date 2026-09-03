# Bracket update rule for the More-Thuente line search (Section 2/Section 3 of
# Moré & Thuente, 1994).
#
# The paper presents the same three-case update twice: as Cases U1-U3
# acting on ψ-values in Section 2 (p. 291), and equivalently as Cases a-c
# acting on φ-values in Section 3 (p. 297). The case structure is identical —
# only the function feeding the tests changes. We implement it once,
# agnostic to whether the inputs are φ-values or ψ-shifted values.
# Below we refer to the cases by their Section 2 names (U1-U3); a-c is the
# same algorithm applied to φ rather than ψ.
#
# Notation in this file matches the paper:
#   αₗ — current best step  (lowest value seen, "anchor")
#   αᵤ — other endpoint of the interval of uncertainty
#   αₜ — current trial step  (the one whose data we just computed)
#
# Note that the paper does not assume αₗ < αᵤ; the interval of
# uncertainty is [min(αₗ, αᵤ), max(αₗ, αᵤ)].

"""
    update_bracket(αₗ, fₗ, gₗ, αᵤ, fᵤ, gᵤ, αₜ, fₜ, gₜ, bracketed)
        -> (αₗ⁺, fₗ⁺, gₗ⁺, αᵤ⁺, fᵤ⁺, gᵤ⁺, bracketed⁺)

Update the interval of uncertainty after evaluating the trial value αₜ.
Implements the three-case algorithm of Moré-Thuente (1994), Section 2
(Cases U1-U3) — equivalently Section 3 (Cases a-c), which is the same case
analysis applied to φ rather than ψ.

# Cases

- **U1 — trial value rose** (`fₜ > fₗ`): a minimizer is bracketed in
  the interval (αₗ, αₜ). Update: `αₗ⁺ = αₗ`, `αᵤ⁺ = αₜ`.

- **U2 — trial fell, slope away from αₗ** (`fₜ ≤ fₗ` and
  `gₜ·(αₗ - αₜ) > 0`): the function continues to descend past αₜ in the
  direction away from αₗ. The minimum lies beyond αₜ; move αₗ forward.
  Update: `αₗ⁺ = αₜ`, `αᵤ⁺ = αᵤ`.

- **U3 — trial fell, slope toward αₗ** (`fₜ ≤ fₗ` and
  `gₜ·(αₗ - αₜ) < 0`): the slope at αₜ points back at αₗ, so a minimizer
  is bracketed in (αₗ, αₜ). αₜ becomes the new best; the old αₗ becomes
  the other endpoint. Update: `αₗ⁺ = αₜ`, `αᵤ⁺ = αₗ`.

# Bracketing

The `bracketed` flag tracks whether αᵤ is meaningful (the interval has
a finite right endpoint). Cases U1 and U3 establish the bracket; Case
U2 extends a still-open interval. Once `bracketed == true`, it stays
true.

# Termination corner case

The paper notes (p. 291) that if `gₜ = 0` and `fₜ ≤ fₗ`, then αₜ already
satisfies `T(μ)` and no update is needed — the caller should detect this
and terminate. This function does not test for it; the `gₜ·(αₗ - αₜ) = 0`
boundary falls into the U3 branch here, which is harmless if the caller
didn't already exit.

# Precondition

The caller must maintain the Section 2 endpoint invariants (paper eq. 2.1):
`fₗ ≤ fᵤ`, `fₗ ≤ 0` (for ψ-values), and `gₗ·(αᵤ - αₗ) < 0`. The paper
proves these are preserved across calls (modulo the `gₜ = 0`
termination case).
"""
function update_bracket(αₗ, fₗ, gₗ, αᵤ, fᵤ, gᵤ, αₜ, fₜ, gₜ, bracketed)
    if fₜ > fₗ
        # Case U1: trial rose. Old αₗ stays; new αᵤ is αₜ.
        return αₗ, fₗ, gₗ, αₜ, fₜ, gₜ, true
    elseif gₜ * (αₗ - αₜ) > zero(gₜ)
        # Case U2: trial fell, slope still pointing away from αₗ.
        # Bracket is not yet (re)established; αᵤ inherits the prior value.
        return αₜ, fₜ, gₜ, αᵤ, fᵤ, gᵤ, bracketed
    else
        # Case U3: trial fell, slope points back toward αₗ.
        # The old αₗ becomes the upper endpoint.
        return αₜ, fₜ, gₜ, αₗ, fₗ, gₗ, true
    end
end
