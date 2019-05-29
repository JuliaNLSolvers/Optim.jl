function assess_convergence(state::IPNewtonState, d, options::Options)
    # We use the whole bstate-gradient `bgrad`
    bgrad = state.bgrad
    Optim.assess_convergence(state.x,
                       state.x_previous,
                       state.L,
                       state.L_previous,
                       [state.g; bgrad.slack_x; bgrad.slack_c; bgrad.λx; bgrad.λc; bgrad.λxE; bgrad.λcE],
                       options.x_abstol,
                       options.f_reltol,
                       options.g_abstol)
end
