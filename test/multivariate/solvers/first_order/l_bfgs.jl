@testset "L-BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric",)

    if Sys.WORD_SIZE == 32
        # Not sure why it doesn't converge; I don't have a system to check it out on
        convergence_exceptions = (("Extended Powell", 1),
                                  ("Extended Powell", 2))
    else
        convergence_exceptions = ()
    end

    @show skip
    @show convergence_exceptions
    run_optim_tests(LBFGS(),
                    f_increase_exceptions = ("Extended Rosenbrock",),
                    skip=skip,
                    convergence_exceptions = convergence_exceptions,
                    show_name = debug_printing,
                    show_res = debug_printing,
                    )
end
