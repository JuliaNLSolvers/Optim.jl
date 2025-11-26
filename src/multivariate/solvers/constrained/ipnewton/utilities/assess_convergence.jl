function assess_convergence(state::IPNewtonState, d, options::Options)
    # We use the whole bstate-gradient `bgrad`
    bgrad = state.bgrad
    assess_convergence(
        state.x,
        state.x_previous,
        state.L_x,
        state.L_x_previous,
        [state.g_L_x; bgrad.slack_x; bgrad.slack_c; bgrad.λx; bgrad.λc; bgrad.λxE; bgrad.λcE],
        options.x_abstol,
        options.f_reltol,
        options.g_abstol,
    )
end
