@testset "N-GMRES" begin
    run_optim_tests(NGMRES();
                    f_increase_exceptions = ("Rosenbrock", "Powell", "Himmelblau"))

end

@testset "O-ACCEL" begin
    run_optim_tests(OACCEL();
                    f_increase_exceptions = ("Rosenbrock", "Hosaki",
                                             "Powell", "Himmelblau"))
end
