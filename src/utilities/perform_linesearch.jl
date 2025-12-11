# Reset the search direction if it becomes corrupted
# return true if the direction was changed
reset_search_direction!(state, d, method) = false # no-op

_alphaguess(a) = a
_alphaguess(a::Number) = LineSearches.InitialStatic(alpha = a)

function reset_search_direction!(state::BFGSState, method::BFGS)
    gx = state.g_x
    if method.initial_invH === nothing
        n = length(state.x)
        T = typeof(state.invH)
        if method.initial_stepnorm === nothing
            state.invH .= _init_identity_matrix(state.x)
        else
            initial_scale = method.initial_stepnorm * inv(norm(gx, Inf))
            state.invH .= _init_identity_matrix(state.x, initial_scale)
        end
    else
        state.invH .= method.initial_invH(state.x)
    end
    #    copyto!(state.invH, method.initial_invH(state.x))
    state.s .= .-gx
    return true
end

function reset_search_direction!(state::LBFGSState, ::LBFGS)
    state.pseudo_iteration = 1
    state.s .= .-state.g_x
    return true
end

function reset_search_direction!(state::ConjugateGradientState, ::ConjugateGradient)
    state.s .= .-state.pg
    return true
end

function perform_linesearch!(state, method, d)
    # Calculate search direction dphi0
    dphi_0 = real(dot(state.g_x, state.s))
    # reset the direction if it becomes corrupted
    if dphi_0 >= zero(dphi_0) && reset_search_direction!(state, method)
        dphi_0 = real(dot(state.g_x, state.s)) # update after direction reset
    end
    phi_0 = state.f_x

    # Guess an alpha
    method.alphaguess!(method.linesearch!, state, phi_0, dphi_0, d)

    # Perform line search; catch LineSearchException to allow graceful exit
    lssuccess = try
        state.alpha, ϕalpha =
            method.linesearch!(d, state.x, state.s, state.alpha, state.x_ls, phi_0, dphi_0)
        true
    catch ex
        if ex isa LineSearches.LineSearchException
            state.alpha = ex.alpha
            # We shouldn't warn here, we should just carry it to the output
            # @warn("Linesearch failed, using alpha = $(state.alpha) and
            # exiting optimization.\nThe linesearch exited with message:\n$(ex.message)")
            false
        else
            rethrow()
        end
    end

    # Store current x and f(x) for next iteration
    state.f_x_previous = phi_0
    copyto!(state.x_previous, state.x)
    if hasproperty(state, :g_x_previous)
        copyto!(state.g_x_previous, state.g_x)
    end

    return lssuccess
end
