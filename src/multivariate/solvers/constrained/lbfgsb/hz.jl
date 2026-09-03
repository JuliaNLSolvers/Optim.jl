# Hager-Zhang Approximate Wolfe line search
#
# Adapted from LineSearches.jl (JuliaNLSolvers/LineSearches.jl) hagerzhangls.jl
# Renamed HagerZhangLS → HZAW to match local naming convention.
#
# TODO
# Original paper implements:
# V1: only Wolfe conditions
# V2: only approximate Wolfe conditions
# V3: initially try to satisfy Wolfe conditions. If the following condition is satisfied for some
#     k then only check the approximate Wolfe conditions for the rest of the iterations:
#  abs(f(x_k+1)-f(x_k)) ≤ ω*Ck
# Qk = 1 + Q_k-1 * Δ, Q_-1 = 0
# Ck = Ck-1 + (abs(fx_k))-Ck-1)/Qk, C_-1 = 0
#
# for Δ ∈ [0,1) and ω ∈ [0,1]. This is all discussed on [p. 121, CG_DESCENT_851].
#
# There is also inital(k) on [p.124, CG_DESCENT_851], but this is quite complicated and
# reaches into the surrounding code and would require a greater line search state. It's
# implemented in the other HagerZhang

"""
    HZAW

An object that controls the Hager-Zhang approximate Wolfe line search
algorithm.[^HZ2005]

    HZAW(; kwargs...)

The `HZAW` constructor takes the following keyword arguments. Default values
correspond to those used in section 5 of [^HZ2005].

 - `decrease`: parameter between 0 and 1, less than or equal to `curvature`,
   specifying sufficient decrease in the objective per the Armijo rule.
   Defaults to `0.1`.
 - `curvature`: parameter between 0 and 1, greater than or equal to `decrease`,
   specifying sufficient decrease in the gradient per the curvature condition.
   Defaults to `0.9`.
 - `theta`: parameter between 0 and 1 that controls the bracketing interval
   update. Defaults to `0.5`, which indicates bisection. (See step U3 of the
   interval update procedure in section 4 of [^HZ2005].)
 - `gamma`: factor by which the length of the bracketing interval should
   decrease at each iteration of the algorithm. Defaults to `2/3`. If such
   a decrease is not achieved, the interval is bisected instead of using the
   output of the secant² step.
 - `epsilon_k`: parameter that controls the approximate Wolfe conditions.
   Defaults to `1e-6`. See [p. 122, CG_DESCENT_851] for more details.
 - `maxiter`: maximum number of iterations to perform in the main loop of
   the algorithm. Defaults to `50`.
 - `maxiter_U3`: maximum number of iterations to perform in step U3 of
   the interval update procedure. Defaults to `50`.
 - `maxiter_finite_check`: maximum number of backtracking iterations to find a
   finite function value from a non-finite initial step length. Defaults to `100`.
 - `rho`: expansion factor used in the `bracket` procedure when searching for
   an interval satisfying the opposite slope condition. Defaults to `5.0`.
 - `rho_finite_check`: contraction factor used to backtrack from a non-finite initial step
   length into a feasible region. Defaults to `1/10`.

We tweak the original algorithm slightly, by backtracking into a feasible
region if the original step length results in function values that are not
finite. This allows us to set up an interval from this point that satisfies
the `bracket` procedure (bottom of [p. 123, CG_DESCENT_851]).

[^HZ2005]: Hager, W. W., & Zhang, H. (2005). A New Conjugate Gradient Method
           with Guaranteed Descent and an Efficient Line Search. SIAM Journal
           on Optimization, 16(1), 170–192. doi:10.1137/030601880
[^CG_DESCENT_851]: Hager, W. W., & Zhang, H. (2006). Algorithm 851: CG_DESCENT,
                   a Conjugate Gradient Method with Guaranteed Descent. ACM
                   Transactions on Mathematical Software, 32(1), 113–137.
                   doi:10.1145/1139480.1139484
"""
struct HZAW{T,Tepsilon} <: LineSearcher
    decrease::T
    curvature::T
    θ::T
    γ::T
    ϵ::Tepsilon
    maxiter::Int
    maxiter_U3::Int
    maxiter_finite_check::Int
    ρ::T
    ρ_finite_check::T
end

Base.summary(::HZAW) = "Approximate Wolfe Line Search (Hager & Zhang)"
HZAW{T}(hzl::HZAW) where {T} = HZAW(
    T(hzl.decrease),
    T(hzl.curvature),
    T(hzl.θ),
    T(hzl.γ),
    T(hzl.ϵ),
    hzl.maxiter,
    hzl.maxiter_U3,
    hzl.maxiter_finite_check,
    T(hzl.ρ),
    T(hzl.ρ_finite_check),
)
function HZAW(;
    decrease = 0.1,
    curvature = 0.9,
    theta = 0.5,
    gamma = 2 / 3,
    maxiter = 50,
    maxiter_U3 = 50,
    maxiter_finite_check = 100,
    epsilon_k = 1e-6,
    rho = 5.0,
    rho_finite_check = 1/10,
)
    if !(0 < decrease ≤ curvature)
        throw(
            ArgumentError(
                "Decrease constant must be positive and smaller than the curvature condition. Got decrease = $decrease and curvature = $curvature",
            ),
        )
    end
    if decrease >= 1/2
        throw(
            ArgumentError(
                "Decrease constant must be smaller than 1/2. Got decrease = $decrease",
            ),
        )
    end
    if curvature >= 1
        throw(
            ArgumentError(
                "Curvature constant must be smaller than one. Got curvature = $curvature",
            ),
        )
    end
    HZAW(
        decrease,
        curvature,
        theta,
        gamma,
        epsilon_k,
        maxiter,
        maxiter_U3,
        maxiter_finite_check,
        rho,
        rho_finite_check,
    )
end

struct TrialBundle{T}
    p::T
    φ::T
    dφ::T
end
function TrialBundle(c, t::Tuple)
    TrialBundle(promote(c, t[1], t[2])...)
end
Base.isfinite(tb::TrialBundle) = isfinite(tb.φ) && isfinite(tb.dφ)

# At the core of this line search we have the (approximate) Wolfe conditions.
struct WolfeSetup{T}
    φ0::T
    dφ0::T
    δ::T
    σ::T
    ϵ::T # [p.182, HZ2005] uses |f(x_k)| see also [p. 122, CG_DESCENT_851] but we do not implement Ck. could rename this to epsilon_rel and have an abs as well.
end
function WolfeSetup(Σ0::TrialBundle, δ, σ, ϵ)
    WolfeSetup(Σ0.φ, Σ0.dφ, δ, σ, ϵ)
end

# [eq (22), p.120, CG_DESCENT_851]
function is_wolfe(wc::WolfeSetup, Σc::TrialBundle)
    (; φ0, dφ0, δ, σ) = wc
    φc, dφc, c = Σc.φ, Σc.dφ, Σc.p
    # Satisfies T1
    δ * dφ0 ≥ (φc - φ0) / c && dφc ≥ σ * dφ0
end
# [eq (23), p.120, CG_DESCENT_851]
function is_approx_wolfe(awc::WolfeSetup, Σc::TrialBundle)
    (; φ0, dφ0, δ, σ, ϵ) = awc
    φc, dφc, c = Σc.φ, Σc.dφ, Σc.p
    # Satisfies T2 and eqn (27) [p. 122, CG_DESCENT_851]
    (2 * δ - 1) * dφ0 ≥ dφc ≥ σ * dφ0 && φc ≤ φ0 + ϵ * abs(φ0)
end

# Armijo sufficient decrease check (no curvature condition)
function is_armijo(wc::WolfeSetup, Σc::TrialBundle)
    (; φ0, dφ0, δ) = wc
    φc, c = Σc.φ, Σc.p
    φc ≤ φ0 + δ * c * dφ0
end

# Roughly equivalent to L0-L3 but we add L0' where
# L0': Check that the right side of the interval yields finite function values and directional derivative values.
#
# αmax: maximum step length (e.g., distance to nearest bound along d).
#   If bracket expansion hits αmax without forming a valid bracket,
#   we fall back to Armijo backtracking and return wolfe=false.
function find_steplength(hzl::HZAW, φ, φ0, dφ0, c::T; αmax::T = T(Inf)) where {T}
    hzl = HZAW{T}(hzl)
    # c = initial(k) but this is done outside
    δ = hzl.decrease
    σ = hzl.curvature
    ρ = hzl.ρ
    ρ_finite_check = hzl.ρ_finite_check
    ϵ = hzl.ϵ
    Σ0 = TrialBundle(T(0), φ0, dφ0)
    if !isfinite(Σ0)
        # Non-finite value/slope at α = 0: nothing to search along. The caller
        # (the L-BFGS-B loop) detects the non-finite α and falls back.
        return T(NaN), T(NaN), false
    end

    Σc = TrialBundle(c, φ(c))
    # Backtrack into feasible region; not part of original algorithm
    iter = 0
    while !isfinite(Σc) && iter <= hzl.maxiter_finite_check
        iter += 1
        # don't use interpolation, this is vanilla backtracking
        c = c*ρ_finite_check
        Σc = TrialBundle(c, φ(c))
    end
    if iter > hzl.maxiter_finite_check
        # Could not backtrack into a region with finite values; signal failure.
        return T(NaN), T(NaN), false
    end
    wolfesetup = WolfeSetup(Σ0, δ, σ, ϵ)

    # Wolfe conditions
    is_wolfe(wolfesetup, Σc) && return Σc.p, Σc.φ, true
    # Approximate Wolfe conditions
    is_approx_wolfe(wolfesetup, Σc) && return Σc.p, Σc.φ, true
    # Set up interval
    Σaj, Σbj, wolfe_in_bracket, hit_αmax = bracket(hzl, Σ0, Σc, φ, ρ, wolfesetup, αmax)
    if wolfe_in_bracket
        return Σaj.p, Σaj.φ, true
    end
    if hit_αmax
        # Bracket expansion reached αmax without forming an opposite-slope bracket.
        # Σaj holds the best point found (sufficient decrease, descent direction).
        # Fall back to Armijo backtracking from this point.
        Σbest = Σaj
        α = Σbest.p
        # Backtrack by halving until Armijo is satisfied (it likely already is)
        for _ = 1:50
            is_armijo(wolfesetup, Σbest) && return Σbest.p, Σbest.φ, false
            α = α / 2
            Σbest = TrialBundle(α, φ(α))
        end
        # Last resort: return whatever we have
        return Σbest.p, Σbest.φ, false
    end

    for j = 1:hzl.maxiter
        # === Step L1: Secant² update ===
        # This is the main step. It's called so because we expect two secant updates to
        # update each end of the interval. First a secant step will move one endpoint,
        # then the other endpoint will be updated by the second secant step. An exception
        # can be that one secant step ends outside of the interval.
        Σa, Σb, iswolfe = secant²(hzl, φ, φ0, Σaj, Σbj, ϵ, wolfesetup)
        if iswolfe
            return Σa.p, Σa.φ, true
        end

        # === Step L2: Bisection if insufficient decrease ===
        # When the interval was not decreasing by at least a factor of γ, we bisect instead.
        # Notice that this forces us not to be in U0 so the interval *will* change.
        aj, bj = Σaj.p, Σbj.p
        a, b = Σa.p, Σb.p
        Σaj, Σbj = if b - a > hzl.γ * (bj - aj)
            c = (a + b) / 2
            update(hzl, Σa, Σb, c, φ, φ0, ϵ)
        else
            Σa, Σb
        end

        # The Wolfe conditions are sufficient but can be hard to satisfy.
        a_wolfe = is_wolfe(wolfesetup, Σaj) || is_approx_wolfe(wolfesetup, Σaj)
        b_wolfe = is_wolfe(wolfesetup, Σbj) || is_approx_wolfe(wolfesetup, Σbj)
        if a_wolfe && b_wolfe
            aj, φaj = Σaj.p, Σaj.φ
            bj, φbj = Σbj.p, Σbj.φ
            if φaj < φbj
                return aj, φaj, true
            else
                return bj, φbj, true
            end
        elseif a_wolfe
            return Σaj.p, Σaj.φ, true
        elseif b_wolfe
            return Σbj.p, Σbj.φ, true
        end
    end
    return T(NaN), T(NaN), false
end

"""
   update_U3_a_c

Used to take step U3 of the updating procedure [p.123, CG_DESCENT_851]. The other steps
are in update, but this step is separated out to be able to use it in
step B2 of bracket. Initialization of ā and b̄ is done outside this call.
"""
function update_U3_a_c(
    hzl::HZAW,
    φ,
    φ0,
    Σā::TrialBundle{T},
    Σb̄::TrialBundle{T},
    ϵ,
) where {T}
    # verified against paper description [p. 123, CG_DESCENT_851]
    θ = hzl.θ

    for j = 1:hzl.maxiter_U3
        # === Step U3.a === convex combination of ā and b̄; 0.5 implies bisection
        ā, b̄ = Σā.p, Σb̄.p
        d = (1 - θ) * ā + θ * b̄
        Σd = TrialBundle(d, φ(d))

        if Σd.dφ ≥ T(0)
            # found point of increasing objective; return with upper bound d
            return Σā, Σd
        else # now Σd.dφ < T(0)
            if Σd.φ ≤ φ0 + ϵ * abs(φ0)
                # === Step U3.b ===
                Σā = Σd
            else # φ(d) ≥ φ0 + ϵ * abs(φ0)
                # === Step U3.c ===
                Σb̄ = Σd
            end
        end
    end
    # Reached the U3 iteration cap without a point of increasing objective;
    # return the current endpoints and let the outer loop proceed.
    return Σā, Σb̄
end

in_bounds(c, Σa, Σb) = Σa.p <= c <= Σb.p

# Full update: bounds check + evaluate + U1-U3. Used by L2 bisection.
function update(hzl::HZ, Σa, Σb, c::T, φ, φ0, ϵ) where {HZ<:HZAW,T}
    # === Step U0: Check c is interior to interval ===
    if !in_bounds(c, Σa, Σb)
        return Σa, Σb
    end
    Σc = TrialBundle(c, φ(c))
    _update(hzl, Σa, Σb, Σc, φ, φ0, ϵ)
end

# Inner update with pre-evaluated Σc: U1-U3 only. Used by secant² after Wolfe check.
function _update(hzl::HZ, Σa, Σb, Σc::TrialBundle{T}, φ, φ0, ϵ) where {HZ<:HZAW,T}
    # verified against paper description [p. 123, CG_DESCENT_851]
    # === Step U1: Positive derivative (update upper bound) ===
    if Σc.dφ ≥ T(0)
        return Σa, Σc
    else # Σc.dφ < T(0)
        # === Step U2: Negative derivative with sufficient decrease ===
        if Σc.φ ≤ φ0 + ϵ * abs(φ0)
            return Σc, Σb
        end
        # === Step U3: Negative derivative without sufficient decrease ===
        Σā, Σb̄ = Σa, Σc
        Σa, Σb = update_U3_a_c(hzl, φ, φ0, Σā, Σb̄, ϵ)
        return Σa, Σb
    end
end
"""
  bracket

Find an interval satisfying the opposite slope condition starting from [0, c] [pp. 123-124, CG_DESCENT_851].

Returns `(Σa, Σb, wolfe_found, hit_αmax)`:
- `wolfe_found`: a point satisfying (approx) Wolfe was found during expansion
- `hit_αmax`: bracket expansion reached αmax without forming a valid bracket;
  Σa holds the best descent point with sufficient decrease
"""
function bracket(
    hzl::HZAW,
    Σ0::TrialBundle{T},
    Σc::TrialBundle{T},
    φ,
    ρ,
    wolfesetup,
    αmax,
) where {T}
    # verified against paper description [pp. 123-124, CG_DESCENT_851]
    φ0 = Σ0.φ
    # === Step B0: Initialize bracket search ===
    # Already checked for initial convergence and finite values
    Σcj = Σc

    # Note, we know that dφ(0) < 0 since we're accepted that the current step is in a
    # direction of descent.
    Σci = Σ0

    # ci is a lower bound and cj is an upper bound (candidate)
    # below we test for the following cases:
    # B1: φ is increasing at cj, set [a,b] to [ci,cj]
    # B2: φ is decreasing at cj but function value is sufficiently larger than φ0
    #     use U3 to update. (This is the only case where we call U3 in the whole algorithm.)
    # B3: φ is decreasing at cj and function value is sufficiently smaller than φ0

    maxj = 100
    j = 0
    while j < maxj && Σcj.dφ < T(0)
        j += 1
        if Σcj.φ > Σ0.φ + Σ0.φ * hzl.ϵ # we could collect all condition checks on one type instead of the wolfe and approx wolfe
            # === Step B2: Decreasing derivative without sufficient decrease ===
            # φ is decreasing at cj but function value is sufficiently larger than
            # φ0 so we must have passed a place with increasing φ, use U3 to update.
            Σa, Σb = update_U3_a_c(hzl, φ, φ0, Σ0, Σcj, hzl.ϵ)
            return Σa, Σb, false, false
        end

        # === Step B3: Decreasing derivative with sufficient decrease ===
        # Move lower bound up to cj, expand by factor ρ > 1
        Σci = Σcj

        cj = ρ * Σcj.p
        # Cap expansion at αmax (box constraint boundary)
        if cj ≥ αmax
            if αmax > Σci.p
                # Evaluate at αmax — it may form a bracket or satisfy Wolfe
                Σcj = TrialBundle(αmax, φ(αmax))
                if is_wolfe(wolfesetup, Σcj) || is_approx_wolfe(wolfesetup, Σcj)
                    return Σcj, Σcj, true, false
                end
                if Σcj.dφ ≥ T(0)
                    # αmax gave us an opposite-slope bracket
                    return Σci, Σcj, false, false
                end
                # Still descending at αmax — can't expand further.
                # Σci is the best point: dφ < 0, φ ≤ φ0 + ε (passed B3 check).
                # If αmax point also has sufficient decrease, prefer it (larger step).
                if Σcj.φ ≤ Σ0.φ + Σ0.φ * hzl.ϵ
                    return Σcj, Σcj, false, true
                end
            end
            return Σci, Σci, false, true
        end
        Σcj = TrialBundle(cj, φ(cj))
        # Check if the new point satisfies Wolfe before continuing expansion
        if is_wolfe(wolfesetup, Σcj) || is_approx_wolfe(wolfesetup, Σcj)
            return Σcj, Σcj, true, false
        end
    end
    if j == maxj
        @warn(
            "Failed to find a bracket satisfying the opposite slope condition after $maxj iterations."
        )
    end

    # implicitly Σcj.dφ ≥ T(0) since we exited the loop =>
    # === Step B1: Positive derivative found (opposite slope condition) ===
    # φ is increasing at cj, set b to cj as this is an upper bound,
    # since φ is initially decreasing.
    return Σci, Σcj, false, false
end
function secant(hzl::HZAW, Σa::TrialBundle{T}, Σb::TrialBundle{T}) where {T}
    # verified against paper description [p. 123, CG_DESCENT_851]
    #(a*dφb - b*dφa)/(dφb - dφa)
    # It has been observed that dφa can be very close to dφb,
    # so we avoid taking the difference
    a, dφa, b, dφb = Σa.p, Σa.dφ, Σb.p, Σb.dφ
    sec = a / (1 - dφa / dφb) + b / (1 - dφb / dφa)
    if isnan(sec)
        sec_naive = (a*dφb - b*dφa)/(dφb - dφa)
        return sec_naive
    end
    return sec
end
function secant²(
    hzl::HZAW,
    φ,
    φ0,
    Σa::TrialBundle{T},
    Σb::TrialBundle{T},
    ϵ,
    wolfesetup,
) where {T}
    # verified against paper description [p. 123, CG_DESCENT_851]
    # === Step S1: First secant step ===
    c = secant(hzl, Σa, Σb)
    if !in_bounds(c, Σa, Σb)
        return Σa, Σb, false
    end
    Σc = TrialBundle(c, φ(c))
    if is_wolfe(wolfesetup, Σc) || is_approx_wolfe(wolfesetup, Σc)
        return Σc, Σc, true
    end
    # First update (U1-U3 with pre-evaluated Σc)
    ΣA, ΣB = _update(hzl, Σa, Σb, Σc, φ, φ0, ϵ)
    updated = false
    c̄ = c
    if c == ΣB.p # B == c
        # === Step S2: Second secant with new upper bound ===
        c̄ = secant(hzl, Σb, ΣB)
        updated = true
    elseif c == ΣA.p # A == c
        # === Step S3: Second secant with new lower bound ===
        c̄ = secant(hzl, Σa, ΣA)
        updated = true
    end

    # === Step S4 ===
    if !updated # so !(c==A || c==B from the paper)
        # === Step S4 (variant 2): Return without second secant ===
        return ΣA, ΣB, false
    end
    if !in_bounds(c̄, ΣA, ΣB)
        return ΣA, ΣB, false
    end
    Σc̄ = TrialBundle(c̄, φ(c̄))
    if is_wolfe(wolfesetup, Σc̄) || is_approx_wolfe(wolfesetup, Σc̄)
        return Σc̄, Σc̄, true
    end
    # === Step S4 (variant 1): Update with second secant point ===
    Σā, Σb̄ = _update(hzl, ΣA, ΣB, Σc̄, φ, φ0, ϵ)
    return Σā, Σb̄, false
end
