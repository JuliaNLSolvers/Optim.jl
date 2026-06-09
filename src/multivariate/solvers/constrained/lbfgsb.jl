# Native L-BFGS-B (limited-memory BFGS with box constraints).
#
# Partial source/paper iomplementation, partial translation of some inner
# methods from the Fortran L-BFGS-B 3.0 algorithm of Byrd, Lu, Nocedal & Zhu (1995),
# "A Limited Memory Algorithm for Bound Constrained Optimization",
# SIAM J. Sci. Comput. 16(5):1190-1208.
#
# Unlike Fminbox (a log-barrier wrapper around an unconstrained solver), this
# handles the bounds natively. Each iteration computes a generalized Cauchy
# point (with lazy heap-ordered breakpoints), then minimizes the quadratic model
# over the free variables, then does a line search along the resulting
# direction. The Hessian approximation is kept in the compact form
# B = θI - W M Wᵀ; the middle solves use Cholesky-factored intermediate matrices
# (formt!/formk!/bmv!) and an incrementally maintained free/active set (freev!),
# and the hot paths are in-place.
#
# The line searches live in lbfgsb/ and are self-contained (no LineSearches.jl
# dependency); they share the find_steplength / LineSearcher interface.
# These are essentially re-implementations and needed better support for
# max steplengths. We can adjust this later and upstream

using NLSolversBase: value_gradient!

abstract type LineSearcher end

include("lbfgsb/hz.jl")              # HZAW (Hager-Zhang approximate Wolfe)
include("lbfgsb/linesearches_mt.jl") # MTLS (Moré-Thuente), pulls in mt/*

struct LBFGSB{L<:LineSearcher,T} <: AbstractConstrainedOptimizer
    m::Int               # number of correction pairs kept in memory
    linesearch::L        # vendored line search (HZAW, MTLS, ...)
    stepsize::T          # default trial step handed to the line search
    clip_subspace::Bool  # true: component-wise clamp (Fortran), false: backtracking (paper)
end

"""
# LBFGSB
## Constructor
```julia
LBFGSB(; m::Integer = 10,
         linesearch::LineSearcher = HZAW(),
         stepsize = 1.0,
         clip_subspace::Bool = true)
```
## Description
`LBFGSB` is a native implementation of the limited-memory BFGS algorithm with
box constraints `lb .<= x .<= ub`. It is a bound-constrained optimizer in its
own right (a sibling of `Fminbox`/`SAMIN`), not a wrapper around an
unconstrained method.

Each iteration computes the generalized Cauchy point along the projected
gradient, minimizes the limited-memory quadratic model over the variables that
remain free at the Cauchy point, and performs a line search (capped at the
distance to the nearest active bound) along the resulting search direction. The
compact representation is maintained with Cholesky-factored intermediate
matrices and an incrementally updated free/active set, so the per-iteration cost
is linear in the number of variables for a fixed memory length `m`.

The memory length `m` controls how many `(s, y)` correction pairs are stored.
`linesearch` selects one of the vendored line searches (`HZAW()` or `MTLS()`).
`clip_subspace` chooses between later style component-wise clamping of the
subspace step (`true`) and the original paper's proportional backtracking (`false`).

Gradient-based convergence uses the **projected gradient**, not the plain
gradient: the run stops when `‖x - P(x - g)‖∞ ≤ g_abstol`, where `P` projects
onto the box `[lb, ub]`. This is the correct first-order stationarity measure
for a bound-constrained problem, and it equals `‖g‖∞` only when no bound is
active. The reported `g_residual` (shown as `|g(x)|` in the results summary)
is this projected-gradient norm.

## References
 - Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A Limited Memory
   Algorithm for Bound Constrained Optimization. SIAM Journal on Scientific
   Computing, 16(5), 1190-1208.
"""
function LBFGSB(;
    m::Integer = 10,
    linesearch::LineSearcher = HZAW(),
    stepsize::Real = 1.0,
    clip_subspace::Bool = true,
)
    LBFGSB(Int(m), linesearch, float(stepsize), clip_subspace)
end

Base.summary(io::IO, ::LBFGSB) = print(io, "L-BFGS-B")

# Minimal state used only to report the termination code. It subtypes
# AbstractOptimizerState (not ZerothOrderState) so that the generic
# x_abschange/f_abschange accessors return the real change values: the
# ZerothOrderState methods return NaN, which would mask the x/f-based
# termination codes (SmallObjectiveChange, SmallXChange, ...).
struct LBFGSBState{Tx,Tf,Tfp} <: AbstractOptimizerState
    x::Tx
    f_x::Tf
    x_previous::Tx
    f_x_previous::Tfp
end

# Compact representation of the limited-memory Hessian approximation
# B = θI - W M Wᵀ, with correction pairs in a circular buffer (S, Y) and the
# middle matrix kept as Cholesky factors (J for formt!, wn for formk!).
# Probably need to simplify this, but tried to really not allocate more than 
# the fortran version.
mutable struct CompactLBFGS{T}
    m::Int                # max history size
    S::Matrix{T}          # n × m, correction pairs sₖ
    Y::Matrix{T}          # n × m, correction pairs yₖ
    SᵀS::Matrix{T}        # m × m, SₖᵀSₖ
    SY::Matrix{T}         # m × m, SₖᵀYₖ (lower tri = L, diag = D)
    J::Matrix{T}          # m × m, upper Cholesky factor of T = θSᵀS + LD⁻¹Lᵀ
    work::Vector{T}       # length 2m, scratch for bmv!
    θ::T                  # Hessian scaling θ = yₖᵀyₖ / yₖᵀsₖ
    k::Int                # number of pairs stored
    head::Int             # circular buffer: physical index of oldest column
    wn::Matrix{T}         # 2m × 2m, factored K-matrix for subspace solve
    wn1::Matrix{T}        # 2m × 2m, inner products
    index::Vector{Int}    # length n: free vars (1:nfree), then active vars
    iwhere::Vector{Int}   # length n: 0=free, >0=at bound
    indx2::Vector{Int}    # length n: entering/leaving scratch (also iorder in cauchy)
    nfree::Int            # number of free variables
    iupdat::Int           # total BFGS updates so far
    cnstnd::Bool          # whether the problem has finite bounds
    s::Vector{T}          # length n, scratch for update_compact! (x_new - x)
    y::Vector{T}          # length n, scratch for update_compact! (g_new - g)
    # Pre-allocated workspace (Fortran keeps these in one long work vector).
    xc::Vector{T}         # length n, Cauchy point / trial point (Fortran: z)
    t_bp::Vector{T}       # length n, breakpoint times (Fortran: t)
    p_ws::Vector{T}       # length 2m, W'd accumulator in Cauchy
    c_ws::Vector{T}       # length 2m, W'(xc - x) coefficients
    Mv::Vector{T}         # length 2m, bmv! scratch in Cauchy
    wb::Vector{T}         # length 2m, Wrow! scratch in Cauchy
    r_ws::Vector{T}       # length n, reduced gradient (Fortran: r)
    du::Vector{T}         # length n, subspace direction (Fortran: d)
    d::Vector{T}          # length n, search direction x̂ - x
end

function CompactLBFGS{T}(n::Integer, m::Integer) where {T}
    CompactLBFGS{T}(
        m,
        zeros(T, n, m),
        zeros(T, n, m),
        zeros(T, m, m),
        zeros(T, m, m),
        zeros(T, m, m),
        zeros(T, 2m),
        one(T),
        0,
        1,
        zeros(T, 2m, 2m),
        zeros(T, 2m, 2m),
        collect(1:n),
        zeros(Int, n),
        zeros(Int, n),
        0,
        0,
        false,
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, 2m),
        zeros(T, 2m),
        zeros(T, 2m),
        zeros(T, 2m),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n),
    )
end

function reset_history!(B::CompactLBFGS{T}) where {T}
    B.k = 0
    B.head = 1
    B.θ = one(T)
    B.iupdat = 0
    B.nfree = 0
    B.wn .= 0
    B.wn1 .= 0
    return B
end

# Map logical column j (1 = oldest) to physical column in the circular buffer.
pcol(B::CompactLBFGS, j::Int) = mod(B.head + j - 2, B.m) + 1

# Build T = θ·SᵀS + L·D⁻¹·Lᵀ and Cholesky-factor into B.J (upper triangle).
# L = strict lower triangle of SY, D = Diagonal(diag(SY)). (Fortran `formt`.)
function formt!(B::CompactLBFGS)
    k = B.k
    k == 0 && return
    J, SY, SS = B.J, B.SY, B.SᵀS

    for i = 1:k
        for j = i:k
            ddum = B.θ * SS[i, j]
            for p = 1:(min(i, j)-1)
                ddum += SY[i, p] * SY[j, p] / SY[p, p]
            end
            J[i, j] = ddum
        end
    end

    cholesky!(Symmetric(@view(J[1:k, 1:k]), :U))
    return
end

# Compute p = M·v with M the middle matrix of the compact form, using the
# Cholesky factor in B.J rather than an explicit inverse. (Fortran `bmv`.)
function bmv!(p::AbstractVector, v::AbstractVector, B::CompactLBFGS)
    k = B.k
    k == 0 && return
    T = eltype(p)

    SY = B.SY
    U = LinearAlgebra.UpperTriangular(@view(B.J[1:k, 1:k]))

    # p₂ = (U')⁻¹ (v₂ + L·D⁻¹·v₁)
    for i = 1:k
        s = v[k+i]
        for j = 1:(i-1)
            s += SY[i, j] * v[j] / SY[j, j]
        end
        p[k+i] = s
    end
    ldiv!(U', @view(p[(k+1):2k]))

    # p₁ = D^{-½}·v₁
    for i = 1:k
        p[i] = v[i] / sqrt(SY[i, i])
    end

    # p₂ = U⁻¹·p₂
    ldiv!(U, @view(p[(k+1):2k]))

    # p₁ = -D^{-½}·p₁ + D⁻¹·Lᵀ·p₂
    for i = 1:k
        p[i] = -p[i] / sqrt(SY[i, i])
        s = zero(T)
        for j = (i+1):k
            s += SY[j, i] * p[k+j] / SY[i, i]
        end
        p[i] += s
    end
    return
end

# Extract row b of the implicit W = [Y  θS] into wb (length 2k).
function Wrow!(wb::AbstractVector, B::CompactLBFGS, b::Int)
    k = B.k
    for j = 1:k
        pj = pcol(B, j)
        wb[j] = B.Y[b, pj]
        wb[k+j] = B.θ * B.S[b, pj]
    end
    return
end

# Count entering/leaving variables and rebuild the free/active index set.
# Returns (nenter, ileave, wrk) where wrk = true if formk! needs to run.
# (Fortran `freev`.)
function freev!(B::CompactLBFGS, iter::Int, updatd::Bool)
    n = length(B.iwhere)
    nenter = 0
    ileave = n + 1

    if iter > 0 && B.cnstnd
        for i = 1:B.nfree
            k = B.index[i]
            if B.iwhere[k] > 0
                ileave -= 1
                B.indx2[ileave] = k
            end
        end
        for i = (B.nfree+1):n
            k = B.index[i]
            if B.iwhere[k] <= 0
                nenter += 1
                B.indx2[nenter] = k
            end
        end
    end

    wrk = (ileave < n + 1) || (nenter > 0) || updatd

    B.nfree = 0
    iact = n + 1
    for i = 1:n
        if B.iwhere[i] <= 0
            B.nfree += 1
            B.index[B.nfree] = i
        else
            iact -= 1
            B.index[iact] = i
        end
    end

    return nenter, ileave, wrk
end

# Build and Cholesky-factor the 2col×2col indefinite K-matrix for the subspace
# solve, incrementally updating the inner-product cache wn1. (Fortran `formk`.)
# Returns info: 0 = success, -1/-2 = Cholesky failure.
function formk!(B::CompactLBFGS, nenter::Int, ileave::Int, updatd::Bool)
    n = length(B.iwhere)
    m = B.m
    col = B.k
    col == 0 && return 0
    T = eltype(B.wn)

    if updatd
        if B.iupdat > m
            for jy = 1:(m-1)
                js = m + jy
                for i = 1:(m-jy)
                    B.wn1[jy+i-1, jy] = B.wn1[jy+i, jy+1]
                end
                for i = 1:(m-jy)
                    B.wn1[js+i-1, js] = B.wn1[js+i, js+1]
                end
                for i = 1:(m-1)
                    B.wn1[m+i, jy] = B.wn1[m+i+1, jy+1]
                end
            end
        end

        ipntr = pcol(B, col)

        for jy = 1:col
            jpntr = pcol(B, jy)
            js = m + jy
            temp1 = zero(T)
            temp2 = zero(T)
            temp3 = zero(T)
            for ii = 1:B.nfree
                k1 = B.index[ii]
                temp1 += B.Y[k1, ipntr] * B.Y[k1, jpntr]
            end
            for ii = (B.nfree+1):n
                k1 = B.index[ii]
                temp2 += B.S[k1, ipntr] * B.S[k1, jpntr]
                temp3 += B.S[k1, ipntr] * B.Y[k1, jpntr]
            end
            B.wn1[col, jy] = temp1
            B.wn1[m+col, js] = temp2
            B.wn1[m+col, jy] = temp3
        end

        jpntr = pcol(B, col)
        for i = 1:col
            ipntr2 = pcol(B, i)
            is = m + i
            temp3 = zero(T)
            for ii = 1:B.nfree
                k1 = B.index[ii]
                temp3 += B.S[k1, ipntr2] * B.Y[k1, jpntr]
            end
            B.wn1[is, col] = temp3
        end
        upcl = col - 1
    else
        upcl = col
    end

    for iy = 1:upcl
        ipntr = pcol(B, iy)
        is = m + iy
        for jy = 1:iy
            jpntr = pcol(B, jy)
            js = m + jy
            temp1 = zero(T)
            temp2 = zero(T)
            temp3 = zero(T)
            temp4 = zero(T)
            for ii = 1:nenter
                k1 = B.indx2[ii]
                temp1 += B.Y[k1, ipntr] * B.Y[k1, jpntr]
                temp2 += B.S[k1, ipntr] * B.S[k1, jpntr]
            end
            for ii = ileave:n
                k1 = B.indx2[ii]
                temp3 += B.Y[k1, ipntr] * B.Y[k1, jpntr]
                temp4 += B.S[k1, ipntr] * B.S[k1, jpntr]
            end
            B.wn1[iy, jy] += temp1 - temp3
            B.wn1[is, js] += -temp2 + temp4
        end
    end

    for iy = 1:upcl
        ipntr = pcol(B, iy)
        is = m + iy
        for jy = 1:upcl
            jpntr = pcol(B, jy)
            temp1 = zero(T)
            temp3 = zero(T)
            for ii = 1:nenter
                k1 = B.indx2[ii]
                temp1 += B.S[k1, ipntr] * B.Y[k1, jpntr]
            end
            for ii = ileave:n
                k1 = B.indx2[ii]
                temp3 += B.S[k1, ipntr] * B.Y[k1, jpntr]
            end
            if is <= jy + m
                B.wn1[is, jy] += temp1 - temp3
            else
                B.wn1[is, jy] += -temp1 + temp3
            end
        end
    end

    θ = B.θ
    wn = B.wn
    wn1 = B.wn1

    for iy = 1:col
        is = col + iy
        is1 = m + iy
        for jy = 1:iy
            js = col + jy
            js1 = m + jy
            wn[jy, iy] = wn1[iy, jy] / θ
            wn[js, is] = wn1[is1, js1] * θ
        end
        for jy = 1:(iy-1)
            wn[jy, is] = -wn1[is1, jy]
        end
        for jy = iy:col
            wn[jy, is] = wn1[is1, jy]
        end
        wn[iy, iy] += B.SY[iy, iy]
    end

    C1 = cholesky!(Symmetric(@view(wn[1:col, 1:col]), :U); check = false)
    issuccess(C1) || return -1

    U1 = LinearAlgebra.UpperTriangular(@view(wn[1:col, 1:col]))
    for js = (col+1):2col
        ldiv!(U1', @view(wn[1:col, js]))
    end

    for is = (col+1):2col
        for js = is:2col
            wn[is, js] += dot(@view(wn[1:col, is]), @view(wn[1:col, js]))
        end
    end

    C2 = cholesky!(Symmetric(@view(wn[(col+1):2col, (col+1):2col]), :U); check = false)
    issuccess(C2) || return -2

    return 0
end

# Project x onto the box [lb, ub].
project(x, lb, ub) = clamp.(x, lb, ub)

# Maximum step before hitting a bound along direction d.
function max_steplength(x, d, lb, ub)
    T = eltype(x)
    αmax = T(Inf)
    for i in eachindex(x)
        if d[i] > 0
            αmax = min(αmax, (ub[i] - x[i]) / d[i])
        elseif d[i] < 0
            αmax = min(αmax, (lb[i] - x[i]) / d[i])
        end
    end
    return αmax
end

# Projected gradient ∞-norm, ‖x - P(x - g)‖∞, computed without allocation.
function pg_norm(x, g, lb, ub)
    result = zero(eltype(g))
    for i in eachindex(x, g, lb, ub)
        gi = g[i]
        if gi < 0
            isfinite(ub[i]) && (gi = max(x[i] - ub[i], gi))
        elseif gi > 0
            isfinite(lb[i]) && (gi = min(x[i] - lb[i], gi))
        end
        agi = abs(gi)
        agi > result && (result = agi)
    end
    return result
end

# Min-heap operations for lazy breakpoint ordering. (Fortran `hpsolb`.)
function hpsolb!(t, iorder, n, iheap)
    if iheap == 0
        for k = 2:n
            val = t[k]
            idx = iorder[k]
            i = k
            while i > 1
                j = i >> 1
                if val < t[j]
                    t[i] = t[j]
                    iorder[i] = iorder[j]
                    i = j
                else
                    break
                end
            end
            t[i] = val
            iorder[i] = idx
        end
    end

    n > 1 || return

    out_val = t[1]
    out_idx = iorder[1]
    val = t[n]
    idx = iorder[n]

    i = 1
    while true
        j = 2i
        j <= n - 1 || break
        if t[j+1] < t[j]
            j += 1
        end
        if t[j] < val
            t[i] = t[j]
            iorder[i] = iorder[j]
            i = j
        else
            break
        end
    end
    t[i] = val
    iorder[i] = idx

    t[n] = out_val
    iorder[n] = out_idx
    return
end

# Algorithm CP: generalized Cauchy point. Writes xᶜ into B.xc and c = Wᵀ(xᶜ - x)
# into B.c_ws. Uses lazy heap ordering of the breakpoints. (Fortran `cauchy`.)
function cauchy_point!(B::CompactLBFGS, x, g, lb, ub)
    T = eltype(x)
    n = length(x)
    bk = B.k
    mcols = 2bk
    θ = B.θ

    Mv = @view(B.Mv[1:mcols])
    fill!(Mv, zero(T))
    wb = @view(B.wb[1:mcols])

    t_bp = B.t_bp
    iorder = B.indx2   # reused: freev!/formk! run after cauchy returns

    p = @view(B.p_ws[1:mcols])
    fill!(p, zero(T))
    c = @view(B.c_ws[1:mcols])
    fill!(c, zero(T))

    f′ = zero(T)
    nbreak = 0
    ibkmin = 0
    bkmin = T(Inf)

    for i = 1:n
        gi = g[i]
        if gi < 0
            ti = (x[i] - ub[i]) / gi
        elseif gi > 0
            ti = (x[i] - lb[i]) / gi
        else
            continue
        end

        ti > 0 || continue

        di = -gi
        f′ -= di * di

        for j = 1:bk
            pj = pcol(B, j)
            p[j] += B.Y[i, pj] * di
            p[bk+j] += B.S[i, pj] * di
        end

        if isfinite(ti)
            nbreak += 1
            t_bp[nbreak] = ti
            iorder[nbreak] = i
            if nbreak == 1 || ti < bkmin
                bkmin = ti
                ibkmin = nbreak
            end
        end
    end

    for j = 1:bk
        p[bk+j] *= θ
    end

    if f′ == zero(T)
        copyto!(B.xc, x)
        fill!(c, zero(T))
        return
    end

    bmv!(Mv, p, B)
    f″ = -θ * f′ - dot(p, Mv)
    Δtmin = -f′ / f″
    t_old = zero(T)

    if nbreak == 0
        Δtmin = max(Δtmin, zero(T))
        xc = B.xc
        copyto!(xc, x)
        for i = 1:n
            g[i] == 0 || (xc[i] = clamp(x[i] - Δtmin * g[i], lb[i], ub[i]))
        end
        c .= Δtmin .* p
        return
    end

    nleft = nbreak
    iter = 1

    while true
        if iter == 1
            tj = bkmin
            ibp = iorder[ibkmin]
        else
            if iter == 2
                if ibkmin != nbreak
                    t_bp[ibkmin] = t_bp[nbreak]
                    iorder[ibkmin] = iorder[nbreak]
                end
            end
            hpsolb!(t_bp, iorder, nleft, iter == 2 ? 0 : 1)
            tj = t_bp[nleft]
            ibp = iorder[nleft]
        end

        dt = tj - t_old

        Δtmin < dt && break

        b = ibp
        xᶜ_b = g[b] < 0 ? ub[b] : lb[b]
        z_b = xᶜ_b - x[b]

        c .+= dt .* p
        g_b = g[b]
        Wrow!(wb, B, b)
        bmv!(Mv, c, B)
        wMc = dot(wb, Mv)
        bmv!(Mv, p, B)
        wMp = dot(wb, Mv)
        bmv!(Mv, wb, B)
        wMw = dot(wb, Mv)
        f′ += dt * f″ + g_b^2 + θ * g_b * z_b - g_b * wMc
        f″ += -θ * g_b^2 - 2 * g_b * wMp - g_b^2 * wMw
        p .+= g_b .* wb

        Δtmin = f″ > 0 ? -f′ / f″ : (f′ < 0 ? T(Inf) : zero(T))
        t_old = tj

        nleft -= 1
        nleft > 0 || break

        iter += 1
    end

    Δtmin = max(Δtmin, zero(T))
    t_old += Δtmin
    xc = B.xc
    copyto!(xc, x)
    for i = 1:n
        if g[i] == 0
            xc[i] = x[i]  # avoid Inf*0 = NaN
        else
            xc[i] = clamp(x[i] - t_old * g[i], lb[i], ub[i])
        end
    end
    c .+= Δtmin .* p

    return
end

# Direct primal subspace minimization using the K factorization from formk! and
# the free set from freev!. Reads xc from B.xc, c from B.c_ws; writes the result
# back into B.xc. (Fortran `cmprlb` + `subsm`.)
function subspace_optimize!(B::CompactLBFGS, x, g, lb, ub; clip::Bool = true)
    T = eltype(x)
    θ = B.θ
    nsub = B.nfree
    col = B.k
    xc = B.xc
    c = @view(B.c_ws[1:2col])

    (nsub == 0 || col == 0) && return  # B.xc already holds the Cauchy point

    col2 = 2col
    wv = B.work

    # Reduced gradient r = -θ(xc - x) - g + W·(M·c).
    r = @view(B.r_ws[1:nsub])
    for i = 1:nsub
        k = B.index[i]
        r[i] = -θ * (xc[k] - x[k]) - g[k]
    end

    bmv!(wv, c, B)   # wv[1:2col] = M·c
    for j = 1:col
        pj = pcol(B, j)
        a1 = wv[j]
        a2 = θ * wv[col+j]
        for i = 1:nsub
            k = B.index[i]
            r[i] += B.Y[k, pj] * a1 + B.S[k, pj] * a2
        end
    end

    # wv = W'Z·r
    for j = 1:col
        pj = pcol(B, j)
        temp1 = zero(T)
        temp2 = zero(T)
        for i = 1:nsub
            k = B.index[i]
            temp1 += B.Y[k, pj] * r[i]
            temp2 += B.S[k, pj] * r[i]
        end
        wv[j] = temp1
        wv[col+j] = θ * temp2
    end

    # Solve K⁻¹·wv via the U'EU factorization (E = diag(-I_col, I_col)).
    wv_view = @view(wv[1:col2])
    U = LinearAlgebra.UpperTriangular(@view(B.wn[1:col2, 1:col2]))
    ldiv!(U', wv_view)
    for i = 1:col
        wv[i] = -wv[i]
    end
    ldiv!(U, wv_view)

    # du = (1/θ)r + (1/θ²)Z'W·wv
    du = @view(B.du[1:nsub])
    copyto!(du, r)
    for j = 1:col
        pj = pcol(B, j)
        for i = 1:nsub
            k = B.index[i]
            du[i] += B.Y[k, pj] * wv[j] / θ + B.S[k, pj] * wv[col+j]
        end
    end
    du ./= θ

    # Project onto bounds, writing directly into B.xc (each element read before
    # written at the same index, matching Fortran subsm).
    x̄ = xc

    if clip
        any_bound_hit = false
        for i = 1:nsub
            k = B.index[i]
            x̄[k] = clamp(xc[k] + du[i], lb[k], ub[k])
            if x̄[k] == lb[k] || x̄[k] == ub[k]
                any_bound_hit = true
            end
        end

        if any_bound_hit
            dd_p = zero(T)
            for i in eachindex(x)
                dd_p += (x̄[i] - x[i]) * g[i]
            end
            if dd_p > 0
                α_star = one(T)
                for i = 1:nsub
                    k = B.index[i]
                    if du[i] > 0
                        α_star = min(α_star, (ub[k] - xc[k]) / du[i])
                    elseif du[i] < 0
                        α_star = min(α_star, (lb[k] - xc[k]) / du[i])
                    end
                end
                for i = 1:nsub
                    k = B.index[i]
                    x̄[k] = xc[k] + α_star * du[i]
                end
            end
        end
    else
        α_star = one(T)
        for i = 1:nsub
            k = B.index[i]
            if du[i] > 0
                α_star = min(α_star, (ub[k] - xc[k]) / du[i])
            elseif du[i] < 0
                α_star = min(α_star, (lb[k] - xc[k]) / du[i])
            end
        end
        for i = 1:nsub
            k = B.index[i]
            x̄[k] = xc[k] + α_star * du[i]
        end
    end

    return
end

# Compute s = x_new - x, y = g_new - g into B.s/B.y, then update the compact
# representation if curvature yᵀs > 0. Returns yᵀs (always computed).
function update_compact!(B::CompactLBFGS, x, x_new, g, g_new)
    s = B.s
    y = B.y
    @. s = x_new - x
    @. y = g_new - g
    yts = dot(y, s)

    yts <= 0 && return yts

    k = B.k
    m = B.m

    if k < m
        # Buffer not full: append to the next column (head stays at 1).
        k += 1
        B.k = k
        B.S[:, k] .= s
        B.Y[:, k] .= y

        for j = 1:(k-1)
            B.SY[k, j] = dot(s, @view(B.Y[:, j]))
            B.SᵀS[j, k] = dot(@view(B.S[:, j]), s)
            B.SᵀS[k, j] = B.SᵀS[j, k]
        end
        B.SY[k, k] = yts
        B.SᵀS[k, k] = dot(s, s)
    else
        # Buffer full: circular overwrite of the oldest column.
        for i = 1:(m-1), j = 1:(m-1)
            B.SY[i, j] = B.SY[i+1, j+1]
            B.SᵀS[i, j] = B.SᵀS[i+1, j+1]
        end

        ph = B.head
        B.S[:, ph] .= s
        B.Y[:, ph] .= y
        B.head = mod(ph, m) + 1

        for j = 1:(m-1)
            pj = pcol(B, j)
            B.SY[m, j] = dot(s, @view(B.Y[:, pj]))
            B.SᵀS[j, m] = dot(@view(B.S[:, pj]), s)
            B.SᵀS[m, j] = B.SᵀS[j, m]
        end
        B.SY[m, m] = yts
        B.SᵀS[m, m] = dot(s, s)
    end

    B.θ = dot(y, y) / yts
    B.iupdat += 1

    formt!(B)   # Cholesky-factor T into B.J

    return yts
end

# Trace entry for L-BFGS-B. The reported gradient norm is the projected
# gradient ∞-norm, the natural stationarity measure for a box-constrained run.
function trace_lbfgsb!(tr, iteration, f_x, pgnorm, x, g_x, options, curr_time)
    dt = Dict{String,Any}()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(x)
        dt["g(x)"] = copy(g_x)
    end
    update!(
        tr,
        iteration,
        f_x,
        pgnorm,
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
    )
end

# Callable that evaluates (ϕ(α), ϕ'(α)) along the search direction without
# allocating per call. `x` and `dir` alias the loop's iterate and B.d, which are
# mutated in place, so this is constructed once and reused every iteration.
mutable struct LBFGSBLineFn{D,Tx}
    obj::D
    x::Tx
    dir::Tx
    xα::Tx
end

function (φ::LBFGSBLineFn)(α)
    @. φ.xα = φ.x + α * φ.dir
    fα, gα = value_gradient!(φ.obj, φ.xα)
    return (fα, dot(gα, φ.dir))
end

function optimize(
    f,
    l::AbstractArray,
    u::AbstractArray,
    x0::AbstractArray,
    method::LBFGSB,
    options::Options = Options();
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    if f isa NonDifferentiable
        f = f.f
    end
    od = OnceDifferentiable(f, x0, zero(eltype(x0)); inplace, autodiff)
    optimize(od, l, u, x0, method, options)
end

function optimize(
    f,
    g,
    l::AbstractArray,
    u::AbstractArray,
    x0::AbstractArray,
    method::LBFGSB,
    options::Options = Options();
    inplace::Bool = true,
)
    g! = inplace ? g : (G, x) -> copyto!(G, g(x))
    od = OnceDifferentiable(f, g!, x0, zero(eltype(x0)))
    optimize(od, l, u, x0, method, options)
end

function optimize(
    d::OnceDifferentiable,
    l::AbstractArray,
    u::AbstractArray,
    x0::AbstractArray,
    method::LBFGSB,
    options::Options = Options(),
)
    T = eltype(x0)
    t0 = time()
    (; callback) = options

    n = length(x0)
    (length(l) == n && length(u) == n) ||
        throw(DimensionMismatch("lb, ub and x0 must have the same length."))
    all(i -> l[i] <= u[i], eachindex(l)) ||
        throw(ArgumentError("Each lower bound must be ≤ the corresponding upper bound."))
    all(i -> l[i] <= x0[i] <= u[i], eachindex(x0)) ||
        throw(ArgumentError("Initial x0 is outside the box [lb, ub]."))

    x = project(copy(x0), l, u)
    f_x, _g = value_gradient!(d, x)
    g = copy(_g)

    B = CompactLBFGS{T}(n, method.m)
    B.cnstnd = any(i -> isfinite(l[i]) || isfinite(u[i]), eachindex(x))
    updatd = false

    x_previous = copy(x)
    f_x_previous = oftype(f_x, NaN)
    x_candidate = similar(x)
    g_new = similar(g)

    # x and B.d alias the fields the line search reads; both are mutated in place.
    φ = LBFGSBLineFn(d, x, B.d, similar(x))

    tr = OptimizationTrace{typeof(f_x),typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace
    options.show_trace && print_header(method)

    pgnorm = pg_norm(x, g, l, u)

    x_converged = false
    f_converged = false
    g_converged = pgnorm <= options.g_abstol
    f_increased = false
    ls_failed = false
    stopped_by_time_limit = false
    f_limit_reached = false
    g_limit_reached = false

    iteration = 0
    _time = time()
    tracing && trace_lbfgsb!(tr, iteration, f_x, pgnorm, x, g, options, _time - t0)
    stopped_by_callback = callback !== nothing && callback((; x, f_x, g_x = g, iteration)) == true

    converged = g_converged
    stopped = stopped_by_callback || !isfinite(f_x) || any(!isfinite, g)

    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        # Generalized Cauchy point (writes B.xc, B.c_ws).
        cauchy_point!(B, x, g, l, u)

        # Free/active sets and the K factorization for the subspace solve.
        for i = 1:n
            B.iwhere[i] = (B.xc[i] <= l[i] || B.xc[i] >= u[i]) ? 1 : 0
        end
        nenter, ileave, wrk = freev!(B, iteration - 1, updatd)
        if wrk && B.k > 0
            info = formk!(B, nenter, ileave, updatd)
            if info != 0
                # Cholesky failure: drop the memory and restart from B = I.
                reset_history!(B)
                updatd = false
                cauchy_point!(B, x, g, l, u)
                for i = 1:n
                    B.iwhere[i] = (B.xc[i] <= l[i] || B.xc[i] >= u[i]) ? 1 : 0
                end
                freev!(B, 0, false)
            end
        end

        # Subspace minimization (writes B.xc); search direction d = xc - x.
        subspace_optimize!(B, x, g, l, u; clip = method.clip_subspace)
        @. B.d = B.xc - x

        dφ0 = dot(g, B.d)
        if dφ0 >= 0 && B.k > 0
            # Not a descent direction: restart from B = I and redo the step.
            reset_history!(B)
            updatd = false
            cauchy_point!(B, x, g, l, u)
            for i = 1:n
                B.iwhere[i] = (B.xc[i] <= l[i] || B.xc[i] >= u[i]) ? 1 : 0
            end
            freev!(B, 0, false)
            subspace_optimize!(B, x, g, l, u; clip = method.clip_subspace)
            @. B.d = B.xc - x
            dφ0 = dot(g, B.d)
        end
        if dφ0 >= 0
            ls_failed = true
            break
        end

        αmax = max_steplength(x, B.d, l, u)
        dnorm = norm(B.d)
        boxed = all(i -> isfinite(l[i]) && isfinite(u[i]), eachindex(x))
        α₀ = if B.k == 0 && !boxed && dnorm > 0
            min(one(T) / dnorm, αmax)
        else
            min(T(method.stepsize), αmax)
        end

        α, _, _ = find_steplength(method.linesearch, φ, f_x, dφ0, α₀; αmax = αmax)
        if !isfinite(α)
            α = min(T(1) / 10^4, αmax)
        end

        # Project onto the box. The line search caps α at αmax (the exact step to
        # the nearest bound), so this only matters at the ULP level when a bound
        # is active — but it guarantees every iterate (and the returned x) is
        # feasible rather than a rounding-width outside.
        @. x_candidate = clamp(x + α * B.d, l, u)
        f_new, _gn = value_gradient!(d, x_candidate)
        g_new .= _gn

        if isfinite(f_new)
            yts = update_compact!(B, x, x_candidate, g, g_new)
            updatd = yts > 0
        else
            updatd = false
        end

        copyto!(x_previous, x)
        f_x_previous = f_x
        copyto!(x, x_candidate)   # in place: keeps φ.x valid
        f_x = f_new
        copyto!(g, g_new)

        pgnorm = pg_norm(x, g, l, u)

        x_converged =
            x_abschange(x, x_previous) <= options.x_abstol ||
            x_relchange(x, x_previous) <= options.x_reltol
        f_converged =
            f_abschange(f_x, f_x_previous) <= options.f_abstol ||
            f_relchange(f_x, f_x_previous) <= options.f_reltol
        g_converged = pgnorm <= options.g_abstol
        f_increased = f_x > f_x_previous
        converged = x_converged || f_converged || g_converged

        tracing && trace_lbfgsb!(tr, iteration, f_x, pgnorm, x, g, options, time() - t0)
        if callback !== nothing
            stopped_by_callback = callback((; x, f_x, g_x = g, iteration)) == true
        end

        _time = time()
        stopped_by_time_limit = _time - t0 > options.time_limit
        f_limit_reached =
            options.f_calls_limit > 0 && NLSolversBase.f_calls(d) >= options.f_calls_limit
        g_limit_reached =
            options.g_calls_limit > 0 && NLSolversBase.g_calls(d) >= options.g_calls_limit

        if stopped_by_callback ||
           stopped_by_time_limit ||
           f_limit_reached ||
           g_limit_reached ||
           (f_increased && !options.allow_f_increases) ||
           !isfinite(f_x) ||
           any(!isfinite, g)
            stopped = true
        end
    end

    _time = time()
    Tf = typeof(f_x)
    f_incr_pick = f_increased && !options.allow_f_increases

    stopped_by = (
        f_limit_reached = f_limit_reached,
        g_limit_reached = g_limit_reached,
        h_limit_reached = false,
        time_limit = stopped_by_time_limit,
        callback = stopped_by_callback,
        f_increased = f_incr_pick,
        ls_failed = ls_failed,
        iterations = iteration == options.iterations,
        x_converged = x_converged,
        f_converged = f_converged,
        g_converged = g_converged,
        small_trustregion_radius = false,
    )

    termination_code = _termination_code(
        d,
        pgnorm,
        LBFGSBState(x, f_x, x_previous, f_x_previous),
        stopped_by,
        options,
    )

    return MultivariateOptimizationResults(
        method,
        x0,
        f_incr_pick ? x_previous : x,
        Tf(f_incr_pick ? f_x_previous : f_x),
        iteration,
        Tf(options.x_abstol),
        Tf(options.x_reltol),
        x_abschange(x, x_previous),
        x_relchange(x, x_previous),
        Tf(options.f_abstol),
        Tf(options.f_reltol),
        f_abschange(f_x, f_x_previous),
        f_relchange(f_x, f_x_previous),
        Tf(options.g_abstol),
        pgnorm,
        tr,
        NLSolversBase.f_calls(d),
        NLSolversBase.g_calls(d),
        NLSolversBase.jvp_calls(d),
        NLSolversBase.h_calls(d),
        NLSolversBase.hvp_calls(d),
        options.time_limit,
        _time - t0,
        stopped_by,
        termination_code,
    )
end
