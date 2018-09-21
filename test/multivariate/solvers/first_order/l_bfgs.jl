@testset "L-BFGS" begin
    # Trigonometric gets stuck in a local minimum?
    skip = ("Trigonometric",)

    if Sys.WORD_SIZE == 32
        iteration_exceptions = (("Extended Powell", 2000),)
    else
        iteration_exceptions = ()
    end

    run_optim_tests(LBFGS(),
                    f_increase_exceptions = ("Extended Rosenbrock",),
                    skip=skip,
                    iteration_exceptions = iteration_exceptions,
                    show_name = debug_printing,
                    )
end
