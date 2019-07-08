# Reset the search direction if it becomes corrupted
# return true if the direction was changed
reset_search_direction!(state, d, method) = false # no-op

function reset_search_direction!(state, d, method::BFGS)
    n = length(state.x)
    T = eltype(state.x)

    if method.initial_invH == nothing
        if method.initial_stepnorm == nothing
            state.invH .= Matrix{T}(I, n, n)
        else
            initial_scale = method.initial_stepnorm * inv(norm(gradient(d), Inf))
            state.invH.= Matrix{T}(initial_scale*I, n, n)
        end
    else
        state.invH .= method.initial_invH(state.x)
    end
#    copyto!(state.invH, method.initial_invH(state.x))
    state.s .= .-gradient(d)
    return true
end

function reset_search_direction!(state, d, method::LBFGS)
    state.pseudo_iteration = 1
    state.s .= .-gradient(d)
    return true
end

function reset_search_direction!(state, d, method::ConjugateGradient)
    state.s .= .-state.pg
    return true
end

function perform_linesearch!(state, method, d)
    # Calculate search direction dphi0
    dphi_0 = real(dot(gradient(d), state.s))
    # reset the direction if it becomes corrupted
    if dphi_0 >= zero(dphi_0) && reset_search_direction!(state, d, method)
        dphi_0 = real(dot(gradient(d), state.s)) # update after direction reset
    end
    phi_0  = value(d)

    # Guess an alpha
    method.alphaguess!(method.linesearch!, state, phi_0, dphi_0, d)

    # Store current x and f(x) for next iteration
    state.f_x_previous = phi_0
    copyto!(state.x_previous, state.x)

    # Perform line search; catch LineSearchException to allow graceful exit
    try
        state.alpha, ϕalpha =
            method.linesearch!(d, state.x, state.s, state.alpha,
                               state.x_ls, phi_0, dphi_0)
        return true # lssuccess = true
    catch ex
        if isa(ex, LineSearches.LineSearchException)
            state.alpha = ex.alpha
            # We shouldn't warn here, we should just carry it to the output
            # @warn("Linesearch failed, using alpha = $(state.alpha) and
            # exiting optimization.\nThe linesearch exited with message:\n$(ex.message)")
            return false # lssuccess = false
        else
            rethrow(ex)
        end
    end
end
