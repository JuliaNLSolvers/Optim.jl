function perform_linesearch{M}(state, method::M, d)
    # Determine the distance of movement along the search line
    if M <: Union{BFGS, Newton} && method.resetalpha == true
        state.alpha = one(state.alpha)
    end
    try
        state.alpha, f_update, g_update =
        method.linesearch!(d, state.x, state.s, state.x_ls, state.g_ls, state.lsr,
                           state.alpha, state.mayterminate)
        state.f_calls, state.g_calls = state.f_calls + f_update, state.g_calls + g_update
        return true
    catch ex
        if isa(ex, LineSearches.LineSearchException)
            state.f_calls, state.g_calls = state.f_calls + ex.f_update, state.g_calls + ex.g_update
            state.alpha = ex.alpha
            Base.warn("Linesearch failed, using alpha = $(state.alpha) and exiting optimization.")
            return false #lssuccess = false
        else
            rethrow(ex)
        end
    end
end
