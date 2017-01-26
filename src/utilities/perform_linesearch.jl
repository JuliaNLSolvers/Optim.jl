perform_linesearch(state, method, d) = perform_linesearch(state, method, d, state.alpha)
function perform_linesearch(state, method, d, alphaguess)
    # Determine the distance of movement along the search line
    try
        state.alpha, f_update, g_update =
        method.linesearch!(d, state.x, state.s, state.x_ls, state.g_ls, state.lsr,
                           alphaguess, state.mayterminate)
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
