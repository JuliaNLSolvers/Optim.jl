#==
What `Newton` asks of a Hessian.

`Newton` uses only generic Base/LinearAlgebra operations on H, so a Hessian can
be any AbstractMatrix that implements them: a dense `Matrix`, a
`SparseMatrixCSC`, or a user-defined type with block, banded, arrowhead, …
structure. A type that implements these efficiently never has an n×n matrix
formed from it:

    similar(H), copyto!(W, H)             current and trial Hessians, and a
                                          workspace W for shifted copies of H
    view(H, diagind(H))                   the diagonal only: bounds on the
                                          spectrum, and shifting by λI in place
    F = cholesky!(W; check = false)       the only solve, in place in a workspace;
    issuccess(F), ldiv!(y, F, b)          issuccess(F) reports whether W is
                                          positive definite
    all(isfinite, H)                      rejecting non-finite Hessians

The generic fallbacks of `cholesky!` and `all(isfinite, H)` are correct for any
AbstractMatrix but read all n² entries (and `cholesky!` does O(n³) work), so a
structured type should specialize `LinearAlgebra.cholesky!(::MyMatrix; check)`
and `Base.all(::typeof(isfinite), ::MyMatrix)`.
==#

#==
The Cholesky factorization of H + λI, computed in the workspace W = similar(H).
Failure to be positive definite is reported through `issuccess`, not thrown.
The factorization may share W's memory, so it is valid only until the next call
with the same W; H itself is never written.
==#
function _shifted_cholesky!(W::AbstractMatrix, H::AbstractMatrix, λ::Real)
    copyto!(W, H)
    iszero(λ) || (view(W, diagind(W)) .+= λ)
    return _cholesky!(W)
end
_cholesky!(W::AbstractMatrix) = cholesky!(W; check = false)
# LAPACK needs `Symmetric` to read one triangle only: a computed Hessian can be
# asymmetric by rounding, which it would reject as not positive definite.
_cholesky!(W::StridedMatrix) = cholesky!(Symmetric(W); check = false)
# CHOLMOD has no in-place factorization of a matrix (its factor is a new object
# anyway), and throws on a matrix that is not exactly symmetric.
_cholesky!(W::AbstractSparseMatrix) = cholesky(Symmetric(W); check = false)

# The smallest diagonal entry. Each Hᵢᵢ = eᵢᵀHeᵢ is a Rayleigh quotient, so
# λ_min(H) ≤ min Hᵢᵢ: no shift τ ≤ -min Hᵢᵢ makes H + τI positive definite.
# (`Base.` since Optim has a `minimum` of its own, for results.)
_min_diag(H::AbstractMatrix) = Base.minimum(view(H, diagind(H)))

#==
Modified Newton: the Cholesky factorization of H + 2τ*I, where τ* is the smallest
τ ≥ 0 in a geometric sequence that makes H + τI positive definite. The search for
τ* is "Cholesky with added multiple of the identity", Algorithm 3.3 in N&W
(2006). It starts at zero when the diagonal is positive, and otherwise just past
-min Hᵢᵢ, below which H + τI is not positive definite (see `_min_diag`).

As τ* ≳ -λ_min(H), the shift H + τ*I is nearly singular along the eigenvector of
the most negative eigenvalue, and the step would be dominated by it. Doubling the
shift instead gives that direction a curvature λ_min + 2τ* ≳ |λ_min|: its
curvature is flipped, as with |H|, which keeps the direction of descent of the
dense factorization this replaces (see issue #153). A positive definite H is not
shifted at all.

The increment β only sets the scale of the shifts, so it is taken relative to the
size of H: N&W use 10⁻³‖H‖_F, and the largest diagonal entry in absolute value
stands in for the norm here, as the diagonal is all the interface reads.

W is a workspace, similar(H). Returns `nothing` if no shift succeeds within
`max_tries` doublings, which happens only for a non-finite H.
==#
function _modified_cholesky!(W::AbstractMatrix, H::AbstractMatrix; max_tries::Int = 64)
    min_d = _min_diag(H)
    scale = Base.maximum(abs, view(H, diagind(H)))
    beta = (iszero(scale) ? one(scale) : scale) / 1000
    tau = min_d > 0 ? zero(beta) : beta - min_d
    for _ = 1:max_tries
        isfinite(tau) || return nothing
        F = _shifted_cholesky!(W, H, tau)
        if issuccess(F)
            return iszero(tau) ? F : _shifted_cholesky!(W, H, 2 * tau)
        end
        tau = max(2 * tau, beta)
    end
    return nothing
end
