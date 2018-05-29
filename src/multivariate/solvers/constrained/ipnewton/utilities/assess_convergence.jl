function assess_convergence(state::IPNewtonState, d, options)
    # We use the whole bstate-gradient `bgrad`
    bgrad = state.bgrad
    Optim.assess_convergence(state.x,
                       state.x_previous,
                       state.L,
                       state.L_previous,
                       [state.g; bgrad.slack_x; bgrad.slack_c; bgrad.λx; bgrad.λc; bgrad.λxE; bgrad.λcE],
                       options.x_tol,
                       options.f_tol,
                       options.g_tol)
end
