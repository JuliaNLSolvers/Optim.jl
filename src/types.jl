abstract Optimizer
abstract ConstrainedOptimizer <: Optimizer
abstract IPOptimizer <: ConstrainedOptimizer
immutable OptimizationOptions{TCallback <: Union{Void, Function}}
    x_tol::Float64
    f_tol::Float64
    g_tol::Float64
    iterations::Int
    store_trace::Bool
    show_trace::Bool
    extended_trace::Bool
    autodiff::Bool
    show_every::Int
    callback::TCallback
    time_limit::Float64
end

function OptimizationOptions(;
        x_tol::Real = 1e-32,
        f_tol::Real = 1e-32,
        g_tol::Real = 1e-8,
        iterations::Integer = 1_000,
        store_trace::Bool = false,
        show_trace::Bool = false,
        extended_trace::Bool = false,
        autodiff::Bool = false,
        show_every::Integer = 1,
        callback = nothing,
        time_limit = NaN)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
        show_trace = true
    end
    OptimizationOptions{typeof(callback)}(
        Float64(x_tol), Float64(f_tol), Float64(g_tol), Int(iterations),
        store_trace, show_trace, extended_trace, autodiff, Int(show_every),
        callback, time_limit)
end

function print_header(options::OptimizationOptions)
    if options.show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
end

function print_header(method::Optimizer)
        @printf "Iter     Function value   Gradient norm \n"
end

immutable OptimizationState{T <: Optimizer}
    iteration::Int
    value::Float64
    g_norm::Float64
    metadata::Dict
end

typealias OptimizationTrace{T} Vector{OptimizationState{T}}

abstract OptimizationResults

type MultivariateOptimizationResults{T,N,M} <: OptimizationResults
    method::String
    initial_x::Array{T,N}
    minimizer::Array{T,N}
    minimum::Float64
    iterations::Int
    iteration_converged::Bool
    x_converged::Bool
    x_tol::Float64
    f_converged::Bool
    f_tol::Float64
    g_converged::Bool
    g_tol::Float64
    trace::OptimizationTrace{M}
    f_calls::Int
    g_calls::Int
    h_calls::Int
end

type UnivariateOptimizationResults{T,M} <: OptimizationResults
    method::String
    initial_lower::T
    initial_upper::T
    minimizer::T
    minimum::Float64
    iterations::Int
    iteration_converged::Bool
    converged::Bool
    rel_tol::Float64
    abs_tol::Float64
    trace::OptimizationTrace{M}
    f_calls::Int
end

immutable NonDifferentiableFunction
    f::Function
end

immutable DifferentiableFunction
    f::Function
    g!::Function
    fg!::Function
end

immutable TwiceDifferentiableFunction
    f::Function
    g!::Function
    fg!::Function
    h!::Function
end

function Base.show(io::IO, t::OptimizationState)
    @printf io "%6d   %14e   %14e\n" t.iteration t.value t.g_norm
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

function Base.show(io::IO, tr::OptimizationTrace)
    @printf io "Iter     Function value   Gradient norm \n"
    @printf io "------   --------------   --------------\n"
    for state in tr
        show(io, state)
    end
    return
end

function Base.show(io::IO, r::MultivariateOptimizationResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" method(r)
    if length(join(initial_state(r), ",")) < 40
        @printf io " * Starting Point: [%s]\n" join(initial_state(r), ",")
    else
        @printf io " * Starting Point: [%s, ...]\n" join(initial_state(r)[1:2], ",")
    end
    if length(join(minimizer(r), ",")) < 40
        @printf io " * Minimizer: [%s]\n" join(minimizer(r), ",")
    else
        @printf io " * Minimizer: [%s, ...]\n" join(minimizer(r)[1:2], ",")
    end
    @printf io " * Minimum: %e\n" minimum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: %s\n" converged(r)
    if r.method == "Nelder-Mead"
        @printf io "   *  √(Σ(yᵢ-ȳ)²)/n < %.1e: %s\n" g_tol(r) g_converged(r)
    else
        @printf io "   * |x - x'| < %.1e: %s\n" x_tol(r) x_converged(r)
        @printf io "   * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" f_tol(r) f_converged(r)
        @printf io "   * |g(x)| < %.1e: %s\n" g_tol(r) g_converged(r)
    end
    @printf io "   * Reached Maximum Number of Iterations: %s\n" iteration_limit_reached(r)
    @printf io " * Objective Function Calls: %d\n" f_calls(r)
    if !(r.method in ("Nelder-Mead", "Simulated Annealing"))
        @printf io " * Gradient Calls: %d" g_calls(r)
    end
    return
end

function Base.show(io::IO, r::UnivariateOptimizationResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" method(r)
    @printf io " * Search Interval: [%f, %f]\n" lower_bound(r) upper_bound(r)
    @printf io " * Minimizer: %e\n" minimizer(r)
    @printf io " * Minimum: %e\n" minimum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(%.1e*|x|+%.1e): %s\n" rel_tol(r) abs_tol(r) converged(r)
    @printf io " * Objective Function Calls: %d" f_calls(r)
    return
end

function Base.append!(a::MultivariateOptimizationResults, b::MultivariateOptimizationResults)
    a.iterations += iterations(b)
    a.minimizer = minimizer(b)
    a.minimum = minimum(b)
    a.iteration_converged = iteration_limit_reached(b)
    a.x_converged = x_converged(b)
    a.f_converged = f_converged(b)
    a.g_converged = g_converged(b)
    append!(a.trace, b.trace)
    a.f_calls += f_calls(b)
    a.g_calls += g_calls(b)
end

# TODO: Expose ability to do forward and backward differencing
function DifferentiableFunction(f::Function)
    function g!(x::Array, storage::Array)
        Calculus.finite_difference!(f, x, storage, :central)
        return
    end
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
    end
    return DifferentiableFunction(f, g!, fg!)
end

function DifferentiableFunction(f::Function, g!::Function)
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
    end
    return DifferentiableFunction(f, g!, fg!)
end

function TwiceDifferentiableFunction(f::Function)
    function g!(x::Vector, storage::Vector)
        Calculus.finite_difference!(f, x, storage, :central)
        return
    end
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    function h!(x::Vector, storage::Matrix)
        Calculus.finite_difference_hessian!(f, x, storage)
        return
    end
    return TwiceDifferentiableFunction(f, g!, fg!, h!)
end

function TwiceDifferentiableFunction(f::Function,
                                     g!::Function,
                                     h!::Function)
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    return TwiceDifferentiableFunction(f, g!, fg!, h!)
end

### Constraints
#
# Constraints are specified by the user as
#    lx_i ≤   x[i]  ≤ ux_i  # variable (box) constraints
#    lc_i ≤ c(x)[i] ≤ uc_i  # linear/nonlinear constraints
# and become equality constraints with l_i = u_i. ±∞ are allowed for l
# and u, in which case the relevant side(s) are unbounded.
#
# The user supplies functions to calculate c(x) and its derivatives.
#
# Of course we could unify the box-constraints into the
# linear/nonlinear constraints, but that would force the user to
# provide the variable-derivatives manually, which would be silly.
#
# This parametrization of the constraints gets "parsed" into a form
# that speeds and simplifies the algorithm, at the cost of many
# additional variables. See `parse_constraints` for details.

immutable ConstraintBounds{T}
    nc::Int          # Number of linear/nonlinear constraints
    # Box-constraints on variables (i.e., directly on x)
    eqx::Vector{Int} # index-vector of equality-constrained x (not actually variable...)
    valx::Vector{T}  # value of equality-constrained x
    ineqx::Vector{Int}  # index-vector of other inequality-constrained variables
    σx::Vector{Int8}    # ±1, in constraints σ(v-b) ≥ 0 (sign depends on whether v>b or v<b)
    bx::Vector{T}       # bound (upper or lower) on variable
    iz::Vector{Int}     # index-vector of nonnegative or nonpositive variables
    σz::Vector{Int8}    # ±1 depending on whether nonnegative or nonpositive
    bz::Vector{T}       # all-zeros, convenience for evaluation of barrier penalty
    # Linear/nonlinear constraint functions and bounds
    eqc::Vector{Int}    # index-vector equality-constrained entries in c
    valc::Vector{T}     # value of the equality-constraint
    ineqc::Vector{Int}  # index-vector of inequality-constraints
    σc::Vector{Int8}    # same as σx, bx except for the nonlinear constraints
    bc::Vector{T}
end
function ConstraintBounds(lx, ux, lc, uc)
    _cb(symmetrize(lx, ux)..., symmetrize(lc, uc)...)
end
function _cb{Tx,Tc}(lx::AbstractArray{Tx}, ux::AbstractArray{Tx}, lc::AbstractVector{Tc}, uc::AbstractVector{Tc})
    T = promote_type(Tx,Tc)
    ConstraintBounds{T}(length(lc), parse_constraints(T, lx, ux, true)..., parse_constraints(T, lc, uc)...)
end

Base.eltype{T}(::Type{ConstraintBounds{T}}) = T
Base.eltype(cb::ConstraintBounds) = eltype(typeof(cb))

nconstraints(cb::ConstraintBounds) = cb.nc

function Base.show(io::IO, cb::ConstraintBounds)
    indent = "    "
    print(io, "ConstraintBounds:")
    print(io, "\n  Variables:")
    showeq(io, indent, cb.eqx, cb.valx, 'x', :bracket)
    showineq(io, indent, cb.ineqx, cb.σx, cb.bx, 'x', :bracket)
    showineq(io, indent, cb.iz, cb.σz, cb.bz, 'x', :bracket)
    print(io, "\n  Linear/nonlinear constraints:")
    showeq(io, indent, cb.eqc, cb.valc, 'c', :subscript)
    showineq(io, indent, cb.ineqc, cb.σc, cb.bc, 'c', :subscript)
    nothing
end

abstract AbstractConstraintsFunction

nconstraints(constraints::AbstractConstraintsFunction) = nconstraints(constraints.bounds)

immutable DifferentiableConstraintsFunction{F,J,T} <: AbstractConstraintsFunction
    c!::F         # c!(x, storage) stores the value of the constraint-functions at x
    jacobian!::J  # jacobian!(x, storage) stores the Jacobian of the constraint-functions
    bounds::ConstraintBounds{T}
end

function DifferentiableConstraintsFunction(c!, jacobian!, lx, ux, lc, uc)
    b = ConstraintBounds(lx, ux, lc, uc)
    DifferentiableConstraintsFunction(c!, jacobian!, b)
end
DifferentiableConstraintsFunction(c!, jacobian!, bounds::ConstraintBounds) =
    DifferentiableConstraintsFunction{typeof(c!), typeof(jacobian!), eltype(b)}(c!, jacobian!, b)

function DifferentiableConstraintsFunction(lx::AbstractArray, ux::AbstractArray)
    bounds = ConstraintBounds(lx, ux, [], [])
    DifferentiableConstraintsFunction(bounds)
end

function DifferentiableConstraintsFunction(bounds::ConstraintBounds)
    c! = (x,c)->nothing
    J! = (x,J)->nothing
    DifferentiableConstraintsFunction(c!, J!, bounds)
end

immutable TwiceDifferentiableConstraintsFunction{F,J,H,T} <: AbstractConstraintsFunction
    c!::F
    jacobian!::J
    h!::H   # Hessian of the barrier terms
    bounds::ConstraintBounds{T}
end
function TwiceDifferentiableConstraintsFunction(c!, jacobian!, h!, lx, ux, lc, uc)
    b = ConstraintBounds(lx, ux, lc, uc)
    TwiceDifferentiableConstraintsFunction(c!, jacobian!, h!, b)
end
TwiceDifferentiableConstraintsFunction(c!, jacobian!, h!, bounds::ConstraintBounds) =
    TwiceDifferentiableConstraintsFunction{typeof(c!), typeof(jacobian!), typeof(h!), eltype(b)}(c!, jacobian!, h!, b)

function TwiceDifferentiableConstraintsFunction(lx::AbstractArray, ux::AbstractArray)
    bounds = ConstraintBounds(lx, ux, [], [])
    TwiceDifferentiableConstraintsFunction(bounds)
end

function TwiceDifferentiableConstraintsFunction(bounds::ConstraintBounds)
    c! = (x,c)->nothing
    J! = (x,J)->nothing
    h! = (x,λ,h)->nothing
    TwiceDifferentiableConstraintsFunction(c!, J!, h!, bounds)
end


## Utilities

function symmetrize(l, u)
    if isempty(l) && !isempty(u)
        l = fill!(similar(u), -Inf)
    end
    if !isempty(l) && isempty(u)
        u = fill!(similar(l), Inf)
    end
    # TODO: change to indices?
    size(l) == size(u) || throw(DimensionMismatch("bounds arrays must be consistent, got sizes $(size(l)) and $(size(u))"))
    _symmetrize(l, u)
end
_symmetrize{T,N}(l::AbstractArray{T,N}, u::AbstractArray{T,N}) = l, u
_symmetrize(l::Vector{Any}, u::Vector{Any}) = _symm(l, u)
_symmetrize(l, u) = _symm(l, u)

# Designed to ensure that bounds supplied as [] don't cause
# unnecessary broadening of the eltype. Note this isn't type-stable; if
# the user cares, it can be avoided by supplying the same concrete
# type for both l and u.
function _symm(l, u)
    if isempty(l) && isempty(u)
        if eltype(l) == Any
            # prevent promotion from returning eltype Any
            l = Array{Union{}}(0)
        end
        if eltype(u) == Any
            u = Array{Union{}}(0)
        end
    end
    promote(l, u)
end

"""
    parse_constraints(T, l, u, split_signed=false) -> eq, val, ineq, σ, b, [iz, σz, bz]

From user-supplied constraints of the form

    l_i ≤  v_i  ≤ u_i

(which include both inequality and equality constraints, the latter
when `l_i == u_i`), convert into the following representation:

    - `eq`, a vector of the indices for which `l[eq] == u[eq]`
    - `val = l[eq] = u[eq]`
    - `ineq`, `σ`, and `b` such that the inequality constraints can be written as
             σ[k]*(v[ineq[k]] - b[k]) ≥ 0
       where `σ[k] = ±1`.
    - optionally (with `split_signed=true`), return an index-vector
      `iz` of entries where one of `l`, `u` is zero, along with
      whether the constraint is `≥ 0` (σz=+1) or `≤ 0` (σz=-1). Such
      are removed from `ineq`, `σ`, and `b`. For coordinate variables
      this can be used to reduce the number of slack variables needed,
      since when one of the bounds is 0, the variable itself *is* a
      slack variable.

Note that since the same `v_i` might have both lower and upper bounds,
`ineq` might have the same index twice (once with `σ`=-1 and once with `σ`=1).

Supplying `±Inf` for elements of `l` and/or `u` implies that `v_i` is
unbounded in the corresponding direction. In such cases there is no
corresponding entry in `ineq`/`σ`/`b`.

T is the element-type of the non-Int outputs
"""
function parse_constraints{T}(::Type{T}, l, u, split_signed::Bool=false)
    size(l) == size(u) || throw(DimensionMismatch("l and u must be the same size, got $(size(l)) and $(size(u))"))
    eq, ineq, iz = Int[], Int[], Int[]
    val, b = T[], T[]
    σ, σz = Array{Int8}(0), Array{Int8}(0)
    for i = 1:length(l)
        li, ui = l[i], u[i]
        li <= ui || throw(ArgumentError("l must be smaller than u, got $li, $ui"))
        if li == ui
            push!(eq, i)
            push!(val, ui)
        else
            if isfinite(li)
                if split_signed && li == 0
                    push!(iz, i)
                    push!(σz, 1)
                else
                    push!(ineq, i)
                    push!(σ, 1)
                    push!(b, li)
                end
            end
            ui = u[i]
            if isfinite(ui)
                if split_signed && ui == 0
                    push!(iz, i)
                    push!(σz, -1)
                else
                    push!(ineq, i)
                    push!(σ, -1)
                    push!(b, ui)
                end
            end
        end
    end
    if split_signed
        return eq, val, ineq, σ, b, iz, σz, zeros(T, length(iz))
    end
    eq, val, ineq, σ, b
end

### Compact printing of constraints

immutable UnquotedString
    str::AbstractString
end
Base.show(io::IO, uqstr::UnquotedString) = print(io, uqstr.str)

Base.array_eltype_show_how(a::Vector{UnquotedString}) = false, ""

if !isdefined(Base, :IOContext)
    IOContext(io; kwargs...) = io
end

function showeq(io, indent, eq, val, chr, style)
    if !isempty(eq)
        print(io, '\n', indent)
        if style == :bracket
            eqstrs = map((i,v) -> UnquotedString("$chr[$i]=$v"), eq, val)
        else
            eqstrs = map((i,v) -> UnquotedString("$(chr)_$i=$v"), eq, val)
        end
        Base.show_vector(IOContext(io, limit=true), eqstrs, "", "")
    end
end

function showineq(io, indent, ineqs, σs, bs, chr, style)
    if !isempty(ineqs)
        print(io, '\n', indent)
        if style == :bracket
            ineqstrs = map((i,σ,b) -> UnquotedString(string("$chr[$i]", ineqstr(σ,b))), ineqs, σs, bs)
        else
            ineqstrs = map((i,σ,b) -> UnquotedString(string("$(chr)_$i", ineqstr(σ,b))), ineqs, σs, bs)
        end
        Base.show_vector(IOContext(io, limit=true), ineqstrs, "", "")
    end
end
ineqstr(σ,b) = σ>0 ? "≥$b" : "≤$b"
