function assess_convergence(x::Array,
                            x_previous::Array,
                            f_x::Real,
                            f_x_previous::Real,
                            g::Array,
                            x_tol::Real,
                            f_tol::Real,
                            g_tol::Real)
    x_converged, f_converged, f_increased, g_converged = false, false, false, false

    if maxdiff(x, x_previous) < x_tol
        x_converged = true
    end

    # Absolute Tolerance
    # if abs(f_x - f_x_previous) < f_tol
    # Relative Tolerance
    if abs(f_x - f_x_previous) < max(f_tol * (abs(f_x) + f_tol), eps(abs(f_x)+abs(f_x_previous)))
        f_converged = true
    end

    if f_x > f_x_previous
        f_increased = true
    end

    if vecnorm(g, Inf) < g_tol
        g_converged = true
    end

    converged = x_converged || f_converged || g_converged

    return x_converged, f_converged, g_converged, converged, f_increased
end


function assess_convergence(state, d, options)
    x_converged, f_converged, f_increased, g_converged = false, false, false, false

    if maxdiff(state.x, state.x_previous) < options.x_tol
        x_converged = true
    end

    # Absolute Tolerance
    # if abs(f_x - f_x_previous) < f_tol
    # Relative Tolerance
    if abs(value(d) - state.f_x_previous) < max(options.f_tol * (abs(value(d)) + options.f_tol), eps(abs(value(d))+abs(state.f_x_previous)))
        f_converged = true
    end

    if value(d) > state.f_x_previous
        f_increased = true
    end

    if vecnorm(gradient(d), Inf) < options.g_tol
        g_converged = true
    end

    converged = x_converged || f_converged || g_converged

    return x_converged, f_converged, g_converged, converged, f_increased
end

function assess_convergence(state::NelderMeadState, d, options)
    g_converged = state.nm_x <= options.g_tol # Hijact g_converged for NM stopping criterior
    return false, false, g_converged, g_converged, false
end


function assess_convergence(state::Union{ParticleSwarmState, SimulatedAnnealingState}, d, options)
    false, false, false, false, false
end



function assess_convergence(state::NewtonTrustRegionState, d, options)
    x_converged, f_converged, g_converged, converged, f_increased = false, false, false, false, false
    if state.rho > state.eta
        # Accept the point and check convergence
        x_converged,
        f_converged,
        g_converged,
        converged,
        f_increased = assess_convergence(state.x,
                                       state.x_previous,
                                       value(d),
                                       state.f_x_previous,
                                       gradient(d),
                                       options.x_tol,
                                       options.f_tol,
                                       options.g_tol)
    end
    x_converged, f_converged, g_converged, converged, f_increased
end
