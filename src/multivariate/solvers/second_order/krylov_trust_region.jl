immutable TwiceDifferentiableHV
    f::Function
    fg!::Function
    hv!::Function
end


immutable KrylovTrustRegion{T <: Real} <: Optimizer
    initial_radius::T
    max_radius::T
    eta::T
    rho_lower::T
    rho_upper::T
    cg_tol::T
end


KrylovTrustRegion(; initial_radius::Real = 1.0,
                    max_radius::Real = 100.0,
                    eta::Real = 0.1,
                    rho_lower::Real = 0.25,
                    rho_upper::Real = 0.75,
                    cg_tol::Real = 0.01) =
                    KrylovTrustRegion(initial_radius, max_radius, eta,
                                  rho_lower, rho_upper, cg_tol)


type KrylovTrustRegionState{T,N,G}
    x::Vector{T}
    g::G
    f_x_previous::T
    x_previous::Vector{T}
    s::Vector{T}
    interior::Bool
    accept_step::Bool
    radius::T
    m_diff::T
    f_diff::T
    rho::T
    r::Vector{T}  # residual vector
    d::Vector{T}  # direction to consider
    Hd::Vector{T} # hessian-vector product
    cg_iters::Int64
end


function initial_state{T}(method::KrylovTrustRegion, options, d, initial_x::Array{T})
    n = length(initial_x)
    # Maintain current gradient in gr
    @assert(method.max_radius > 0)
    @assert(0 < method.initial_radius < method.max_radius)
    @assert(0 <= method.eta < method.rho_lower)
    @assert(method.rho_lower < method.rho_upper)
    @assert(method.rho_lower >= 0)

    g = similar(initial_x)
    f_x = d.fg!(initial_x, g)

    KrylovTrustRegionState("Krylov Trust Region",
                         length(initial_x),
                         copy(initial_x), # Maintain current state in state.x
                         f_x, # Store current f in state.f_x
                         1, # track f calls in state.f_calls
                         1, # track g calls in state.g_calls
                         0,  # track hessian-vector products
                         g, # Store current gradient in state.g
                         0., # f_x_previous
                         similar(initial_x), # x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         true,  # interior
                         true,  # accept step
                         method.initial_radius,
                         0., # model change
                         0., # observed f change
                         0., # state.rho
                         Vector{T}(n),  # residual vector
                         Vector{T}(n),  # direction to consider
                         Vector{T}(n), # hessian-vector product
                         0)
end


function trace!(tr, state, iteration, method::KrylovTrustRegion, options)
    dt = Dict()
    if options.extended_trace
        dt["radius"] = copy(state.radius)
        dt["interior"] = state.interior
        dt["accept_step"] = state.accept_step
        dt["norm(s)"] = norm(state.s)
        dt["rho"] = state.rho
        dt["m_diff"] = state.m_diff
        dt["f_diff"] = state.f_diff
        dt["cg_iters"] = state.cg_iters
    end
    g_norm = norm(state.g, Inf)
    update!(tr,
            iteration,
            state.f_x,
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end


function cg_steihaug!{T}(objective::TwiceDifferentiableHV,
                         state::KrylovTrustRegionState{T},
                         method::KrylovTrustRegion)
    n = length(state.x)
    x, g, d, r, z, Hd = state.x, state.g, state.d, state.r, state.s, state.Hd

    fill!(z, 0.0)  # the search direction is initialized to the 0 vector,
    r[:] = g  # so at first the whole gradient is the residual.
    d[:] = -r # the first direction is the direction of steepest descent.
    rho0 = 1e100  # just a big number

    state.cg_iters = 0
    for i in 1:n
        state.cg_iters += 1
        objective.hv!(x, d, Hd)

        dHd = dot(d, Hd)
        if -1e-15 < dHd < 1e-15
            break
        end

        alpha = dot(r, r) / dHd

        if dHd < 0. || norm(z + alpha * d) >= state.radius
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

    objective.hv!(x, z, Hd)
    return dot(g, z) + 0.5 * dot(z, Hd)
end


function update_state!{T}(objective::TwiceDifferentiableHV,
                          state::KrylovTrustRegionState{T},
                          method::KrylovTrustRegion)
    state.m_diff = cg_steihaug!(objective, state, method)
    @assert state.m_diff <= 0

    state.f_diff = objective.f(state.x + state.s) - state.f_x
    state.rho = state.f_diff / state.m_diff
    state.interior = norm(state.s) < 0.9 * state.radius

    if state.rho < method.rho_lower
        state.radius *= 0.25
    elseif (state.rho > method.rho_upper) && (!state.interior)
        state.radius = min(2 * state.radius, method.max_radius)
    end

    state.accept_step = state.rho > method.eta
    if state.accept_step
        state.x .+= state.s
    end

    return false
end


function update_g!(objective, state::KrylovTrustRegionState, method::KrylovTrustRegion)
    if state.accept_step
        # Update the function value and gradient
        state.f_x_previous, state.f_x = state.f_x, objective.fg!(state.x, state.g)
        state.f_calls, state.g_calls = state.f_calls + 1, state.g_calls + 1
    end
end


function assess_convergence(state::KrylovTrustRegionState, options)
    if !state.accept_step
        return state.radius < options.x_tol, false, false, false, false
    end

    x_converged, f_converged, f_increased, g_converged = false, false, false, false

    if norm(state.s, Inf) < options.x_tol
        x_converged = true
    end

    # Absolute Tolerance
    # if abs(f_x - f_x_previous) < f_tol
    # Relative Tolerance
    if abs(state.f_diff) < max(options.f_tol * (abs(state.f_x) + options.f_tol), eps(abs(state.f_x)+abs(state.f_x_previous)))
        f_converged = true
    end

    if vecnorm(state.g, Inf) < options.g_tol
        g_converged = true
    end

    converged = x_converged || f_converged || g_converged

    return x_converged, f_converged, g_converged, converged, false
end
