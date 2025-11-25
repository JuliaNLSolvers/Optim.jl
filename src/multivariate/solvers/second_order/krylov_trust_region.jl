struct KrylovTrustRegion{T<:Real} <: SecondOrderOptimizer
    initial_radius::T
    max_radius::T
    eta::T
    rho_lower::T
    rho_upper::T
    cg_tol::T
end


KrylovTrustRegion(;
    initial_radius::Real = 1.0,
    max_radius::Real = 100.0,
    eta::Real = 0.1,
    rho_lower::Real = 0.25,
    rho_upper::Real = 0.75,
    cg_tol::Real = 0.01,
) = KrylovTrustRegion(initial_radius, max_radius, eta, rho_lower, rho_upper, cg_tol)

# TODO: support x::Array{T,N} et al.?
mutable struct KrylovTrustRegionState{T} <: AbstractOptimizerState
    x::Vector{T}
    f_x::T
    g_x::Vector{T}
    x_previous::Vector{T}
    f_x_previous::T
    x_cache::Vector{T}
    s::Vector{T}
    interior::Bool
    accept_step::Bool
    radius::T
    m_diff::T
    f_diff::T
    rho::T
    r::Vector{T}  # residual vector
    d::Vector{T}  # direction to consider
    cg_iters::Int
end

function initial_state(method::KrylovTrustRegion, options::Options, d, initial_x::Array{T}) where {T}
    n = length(initial_x)
    # Maintain current gradient in gr
    @assert(method.max_radius > 0)
    @assert(0 < method.initial_radius < method.max_radius)
    @assert(0 <= method.eta < method.rho_lower)
    @assert(method.rho_lower < method.rho_upper)
    @assert(method.rho_lower >= 0)

    f_x, g_x = value_gradient!(d, initial_x)

    KrylovTrustRegionState(
        copy(initial_x),    # Maintain current state in state.x
        f_x,                # Maintain f of current state in state.f_x
        copy(g_x),          # Maintain gradient of current state in state.g_x
        copy(initial_x), # x_previous
        oftype(f_x, NaN),   # f_x_previous
        fill!(similar(initial_x), NaN), # In-place cache for `update_state!`
        fill!(similar(initial_x), NaN), # Maintain current search direction in state.s
        true,               # interior
        true,               # accept step
        convert(T, method.initial_radius),
        zero(T),            # model change
        zero(T),            # observed f change
        zero(T),            # state.rho
        Vector{T}(undef, n),       # residual vector
        Vector{T}(undef, n),       # direction to consider
        0,
    )                  # cg_iters
end


function trace!(
    tr,
    d,
    state::KrylovTrustRegionState,
    iteration::Integer,
    method::KrylovTrustRegion,
    options::Options,
    curr_time = time(),
)
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["radius"] = copy(state.radius)
        dt["interior"] = state.interior
        dt["accept_step"] = state.accept_step
        dt["norm(s)"] = norm(state.s)
        dt["rho"] = state.rho
        dt["m_diff"] = state.m_diff
        dt["f_diff"] = state.f_diff
        dt["cg_iters"] = state.cg_iters
    end
    update!(
        tr,
        iteration,
        state.f_x,
        g_residual(state),
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
        options.callback,
    )
end


function cg_steihaug!(
    objective::TwiceDifferentiable,
    state::KrylovTrustRegionState{T},
    method::KrylovTrustRegion,
) where {T}
    (; x, g_x, d, r) = state
    z = state.s

    fill!(z, 0.0)  # the search direction is initialized to the 0 vector,
    copyto!(r, g_x)  # so at first the whole gradient is the residual.
    d .= .-r # the first direction is the direction of steepest descent.
    rho0 = 1e100  # just a big number

    state.cg_iters = 0
    for i = 1:length(x)
        state.cg_iters += 1
        Hd = hv_product!(objective, x, d)
        dHd = dot(d, Hd)
        if -1e-15 < dHd < 1e-15
            break
        end

        alpha = dot(r, r) / dHd

        if dHd < 0.0 || norm(z .+ alpha .* d) >= state.radius
            a_ = dot(d, d)
            b_ = 2 * dot(z, d)
            c_ = dot(z, z) - state.radius^2
            tau = (-b_ + sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
            z .+= tau .* d
            break
        end

        z .+= alpha .* d
        rho_prev = dot(r, r)
        if i == 1
            rho0 = rho_prev
        end
        r .+= alpha * Hd
        rho_next = dot(r, r)
        r_sqnorm_ratio = rho_next / rho_prev
        d[:] = -r + r_sqnorm_ratio * d

        if (rho_next / rho0) < method.cg_tol^2
            break
        end
    end

    Hd = hv_product!(objective, x, z)
    return dot(g_x, z) + 0.5 * dot(z, Hd)
end


function update_state!(
    objective::TwiceDifferentiable,
    state::KrylovTrustRegionState,
    method::KrylovTrustRegion,
)
    state.m_diff = cg_steihaug!(objective, state, method)
    @assert state.m_diff <= 0

    state.x_cache .= state.x .+ state.s
    f_x_cache = NLSolversBase.value!(objective, state.x_cache)
    state.f_diff = f_x_cache - state.f_x
    state.rho = state.f_diff / state.m_diff
    state.interior = norm(state.s) < 0.9 * state.radius

    if state.rho < method.rho_lower
        state.radius *= 0.25
    elseif (state.rho > method.rho_upper) && (!state.interior)
        state.radius = min(2 * state.radius, method.max_radius)
    end

    state.accept_step = state.rho > method.eta
    if state.accept_step
        # Update history
        copyto!(state.x_previous, state.x)
        state.f_x_previous = state.f_x

        # Update state, function value and its gradient
        copyto!(state.x, state.x_cache)
        state.f_x = f_x_cache
        g_x = gradient!(objective, state.x)
        copyto!(state.g_x, g_x)
    end

    return false
end

# Function value and gradient are already updated in update_state!
update_fgh!(objective, state::KrylovTrustRegionState, ::KrylovTrustRegion) = nothing

function assess_convergence(state::KrylovTrustRegionState, d, options::Options)
    if !state.accept_step
        return state.radius < options.x_abstol, false, false, false
    end

    x_converged, f_converged, f_increased, g_converged = false, false, false, false

    if norm(state.s, Inf) < options.x_abstol
        x_converged = true
    end

    # Absolute Tolerance
    # if abs(f_x - f_x_previous) < f_tol
    # Relative Tolerance
    if abs(state.f_diff) < max(
        options.f_reltol * (abs(state.f_x) + options.f_reltol),
        eps(abs(state.f_x) + abs(state.f_x_previous)),
    )
        f_converged = true
    end

    if norm(state.g_x, Inf) < options.g_abstol
        g_converged = true
    end

    return x_converged, f_converged, g_converged, false
end
