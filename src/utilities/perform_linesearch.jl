#checked_dphi0!(state, d, method) = vecdot(gradient(d), state.s)
checked_dphi0!(state, d, method) = sum(gradient(d).*state.s)
function checked_dphi0!(state, d, method::M) where M<:Union{BFGS, LBFGS}
    # If invH is not positive definite, reset it
    #dphi0 = vecdot(gradient(d), state.s)
    dphi0 = sum(gradient(d).*state.s)
    if dphi0 >= zero(dphi0)
        # "reset" Hessian approximation
        if M <: BFGS
            copy!(state.invH, method.initial_invH(state.x))
        elseif M <: LBFGS
            state.pseudo_iteration = 1
        end

        # Re-calculate direction
        state.s .= .-gradient(d)
        #dphi0 = vecdot(gradient(d), state.s)
        dphi0 = sum(gradient(d).*state.s)
    end
    return dphi0
end
function checked_dphi0!(state, d, method::ConjugateGradient)
    # Reset the search direction if it becomes corrupted
    dphi0 = sum(gradient(d).*state.s)
    if dphi0 >= zero(dphi0)
        state.s .= .-state.pg
        dphi0 = sum(gradient(d).*state.s)
    end
    dphi0
end

function perform_linesearch!(state, method::M, d) where M
    # Calculate search direction dphi0
    dphi0 = checked_dphi0!(state, d, method)
    phi0  = value(d)

    # Refresh the line search cache
    LineSearches.clear!(state.lsr)
    push!(state.lsr, zero(phi0), phi0, dphi0)

    # Guess an alpha
    method.alphaguess!(state, dphi0, d)

    # Store current x and f(x) for next iteration
    state.f_x_previous = phi0
    copy!(state.x_previous, state.x)

    # Perform line search; catch LineSearchException to allow graceful exit
    try
        state.alpha =
            method.linesearch!(d, state.x, state.s, state.x_ls, state.lsr,
                               state.alpha, state.mayterminate)
        state.dphi0_previous = dphi0
        return true # lssuccess = true
    catch ex
        state.dphi0_previous = dphi0
        if isa(ex, LineSearches.LineSearchException)
            state.alpha = ex.alpha
            Base.warn("Linesearch failed, using alpha = $(state.alpha) and exiting optimization.")
            return false # lssuccess = false
        else
            rethrow(ex)
        end
    end
end
