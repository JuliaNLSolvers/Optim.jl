@testset "IPNewton Unconstrained" begin
    method = IPNewton()
    run_optim_tests_constrained(method; show_name=debug_printing, show_res=false, show_itcalls=debug_printing)
end
