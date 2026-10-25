# Reset the search direction if it becomes corrupted
# return true if the direction was changed
reset_search_direction!(state, d, method) = false # no-op
reset_search_direction!(state, method) = false # no-op (2-arg fallback)

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

# A line search can report success without moving x: HagerZhang can return
# alpha = 0, and on a noisy objective a step below the resolution of x (its
# bracket collapses to adjacent floats, or the Armijo decrease of BackTracking
# rounds away). The solver then repeats the same step forever. Count both as
# failures, so the caller resets the search direction and retries once. They are
# reported as a return value rather than thrown, since a throw costs far more than
# an iteration on a cheap objective.
_linesearch_moved_x(state) = state.alpha > zero(state.alpha) && state.x_ls != state.x

function perform_linesearch!(state, method, d)
    # Calculate search direction
    dphi_0 = real(dot(state.g_x, state.s))
    # reset the direction if it becomes corrupted
    if dphi_0 >= zero(dphi_0) && reset_search_direction!(state, method)
        dphi_0 = real(dot(state.g_x, state.s)) # update after direction reset
    end
    phi_0 = state.f_x

    # Guess an alpha
    method.alphaguess!(method.linesearch!, state, phi_0, dphi_0, d)

    # Perform line search; catch LineSearchException to allow graceful exit.
    # A step that does not move x counts as a failure, see `_linesearch_moved_x`.
    #
    # On success `state.x_ls` holds the accepted step, computed exactly as the objective
    # evaluated it (LineSearches >= 7.8). Solvers propose that point rather than
    # recompute `x + alpha*s`, which rounds differently: `update_fgh!` would then evaluate
    # a neighbour the line search never examined, so `accept_step!` could reject a step
    # for a non-finite gradient the vetted point does not have.
    local ϕalpha
    lssuccess = try
        state.alpha, ϕalpha =
            method.linesearch!(d, state.x, state.s, state.alpha, state.x_ls, phi_0, dphi_0)
        _linesearch_moved_x(state)
    catch ex
        if ex isa LineSearches.LineSearchException
            state.alpha = ex.alpha
            false
        else
            rethrow()
        end
    end

    # On failure, reset the search direction and retry once
    if !lssuccess && reset_search_direction!(state, method)
        dphi_0 = real(dot(state.g_x, state.s))
        method.alphaguess!(method.linesearch!, state, phi_0, dphi_0, d)
        lssuccess = try
            state.alpha, ϕalpha =
                method.linesearch!(d, state.x, state.s, state.alpha, state.x_ls, phi_0, dphi_0)
            _linesearch_moved_x(state)
        catch ex2
            if ex2 isa LineSearches.LineSearchException
                state.alpha = ex2.alpha
                false
            else
                rethrow()
            end
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
