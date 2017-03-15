checked_dphi0!(state, d, method) = vecdot(gradient(d), state.s)
function checked_dphi0!{M<:Union{BFGS, LBFGS}}(state, d, method::M)
    # If invH is not positive definite, reset it
    dphi0 = vecdot(gradient(d), state.s)
    if dphi0 > 0.0
        # "reset" Hessian approximation
        if M <: BFGS
            copy!(state.invH, method.initial_invH(state.x))
        elseif M <: LBFGS
            state.pseudo_iteration = 1
        end

        # Re-calculate direction
        @simd for i in 1:state.n
            @inbounds state.s[i] = -gradient(d, i)
        end
        dphi0 = vecdot(gradient(d), state.s)
    end
    return dphi0
end
function checked_dphi0!(state, d, method::ConjugateGradient)
    # Reset the search direction if it becomes corrupted
    dphi0 = vecdot(gradient(d), state.s)
    if dphi0 >= 0
        @simd for i in 1:state.n
            @inbounds state.s[i] = -state.pg[i]
        end
        dphi0 = vecdot(gradient(d), state.s)
    end
    dphi0
end

alphaguess!(state, method, dphi0, d) = nothing
function alphaguess!(state, method::LBFGS, dphi0, d)
    # compute an initial guess for the linesearch based on
    # Nocedal/Wright, 2nd ed, (3.60)
    # TODO: this is a temporary fix, but should eventually be split off into
    #       a separate type and possibly live in LineSearches; see #294
    if method.extrapolate && state.pseudo_iteration > 1
        alphaguess = 2.0 * (value(d) - state.f_x_previous) / dphi0
        alphaguess = max(alphaguess, state.alpha/4.0)  # not too much reduction
        # if alphaguess ≈ 1, then make it 1 (Newton-type behaviour)
        if method.snap2one[1] < alphaguess < method.snap2one[2]
            alphaguess = one(state.alpha)
        end
        state.alpha = alphaguess
    end
end
# Reset to 1 for BFGS and Newton if resetalpha
function alphaguess!(state, method::Union{BFGS, Newton}, dphi0, d)
    if method.resetalpha == true
        state.alpha = one(state.alpha)
    end
end
function alphaguess!(state, method::ConjugateGradient, dphi0, d)
    # Pick the initial step size (HZ #I1-I2)
    state.alpha, state.mayterminate, f_update, g_update =
      LineSearches.alphatry(state.alpha, d, state.x, state.s, state.x_ls, state.g_ls, state.lsr)
    d.f_calls, d.g_calls = d.f_calls + f_update, d.g_calls + g_update
end
function perform_linesearch!{M}(state, method::M, d)
    # Calculate search direction dphi0
    dphi0 = checked_dphi0!(state, d, method)

    # Refresh the line search cache
    LineSearches.clear!(state.lsr)
    push!(state.lsr, zero(value(d)), value(d), dphi0)

    # Guess an alpha
    alphaguess!(state, method, dphi0, d)

    # Perform line search; catch LineSearchException to allow graceful exit
    try
        state.alpha, f_update, g_update =
        method.linesearch!(d, state.x, state.s, state.x_ls, state.g_ls, state.lsr,
                           state.alpha, state.mayterminate)
        # TODO: linesearch now updates d.f_calls through NLSolversBase.value!
        # Should we create a separate property state.f_calls_ls to track the number of ls calls?
        #d.f_calls, d.g_calls = d.f_calls + f_update, d.g_calls + g_update
        return true
    catch ex
        if isa(ex, LineSearches.LineSearchException)
            d.f_calls, d.g_calls = d.f_calls + ex.f_update, d.g_calls + ex.g_update
            state.alpha = ex.alpha
            Base.warn("Linesearch failed, using alpha = $(state.alpha) and exiting optimization.")
            return false #lssuccess = false
        else
            rethrow(ex)
        end
    end
end
