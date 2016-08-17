function assess_convergence(x::Array,
                            x_previous::Array,
                            f_x::Real,
                            f_x_previous::Real,
                            gr::Array,
                            x_tol::Real,
                            f_tol::Real,
                            g_tol::Real)
    x_converged, f_converged, g_converged = false, false, false

    if maxdiff(x, x_previous) < x_tol
        x_converged = true
    end

    # Absolute Tolerance
    # if abs(f_x - f_x_previous) < f_tol
    # Relative Tolerance
    if abs(f_x - f_x_previous) / (abs(f_x) + f_tol) < f_tol || nextfloat(f_x) >= f_x_previous
        f_converged = true
    end

    if vecnorm(gr, Inf) < g_tol
        g_converged = true
    end

    converged = x_converged || f_converged || g_converged

    return x_converged, f_converged, g_converged, converged
end


function assess_convergence(state, options)
    x_converged, f_converged, g_converged = false, false, false

    if maxdiff(state.x, state.x_previous) < options.x_tol
        x_converged = true
    end

    # Absolute Tolerance
    # if abs(f_x - f_x_previous) < f_tol
    # Relative Tolerance
    if abs(state.f_x - state.f_x_previous) / (abs(state.f_x) + options.f_tol) < options.f_tol || nextfloat(state.f_x) >= state.f_x_previous
        f_converged = true
    end

    if vecnorm(state.g, Inf) < options.g_tol
        g_converged = true
    end

    converged = x_converged || f_converged || g_converged

    return x_converged, f_converged, g_converged, converged
end

function assess_convergence(state::NelderMeadState, options)
    g_converged = state.f_x <= options.g_tol # Hijact g_converged for NM stopping criterior
    return false, false, g_converged, g_converged
end


function assess_convergence(state::ParticleSwarmState, options)
    false, false, false, false
end
