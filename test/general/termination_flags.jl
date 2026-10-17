@testset "Termination flags" begin
    TC = Optim.TerminationCode

    @testset "TerminationFlags" begin
        flags = Optim.TerminationFlags(())
        @test isempty(flags)
        @test repr(flags) == "TerminationFlags()"

        flags = Optim.TerminationFlags((TC.Iterations, TC.GradientNorm, TC.Iterations))
        @test length(flags) == 2
        @test TC.GradientNorm in flags
        @test !(TC.SmallXChange in flags)
        @test collect(flags) == [TC.GradientNorm, TC.Iterations]
        @test repr(flags) == "TerminationFlags(GradientNorm, Iterations)"
    end

    rosen(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    x0 = [-1.2, 1.0]

    @testset "multiple criteria" begin
        # Converge in exactly the last allowed iteration
        n = Optim.iterations(optimize(rosen, x0, BFGS(); autodiff = AutoForwardDiff()))
        res = optimize(rosen, x0, BFGS(), Optim.Options(; iterations = n); autodiff = AutoForwardDiff())
        @test collect(Optim.termination_flags(res)) == [TC.GradientNorm, TC.Iterations]
        @test occursin("Termination flags: GradientNorm, Iterations", sprint(show, res))
    end

    @testset "non-finite objective" begin
        res = optimize(x -> NaN, [0.0], NelderMead())
        @test TC.ObjectiveNotFinite in Optim.termination_flags(res)
        @test !Optim.converged(res)
    end

    @testset "failed line search" begin
        # Stalls at the NaN boundary: no change in x, but not converged (#1220)
        res = optimize(x -> x[1] > 0.5 ? NaN : (x[1] - 1)^2, [0.0], BFGS(); autodiff = AutoForwardDiff())
        flags = Optim.termination_flags(res)
        @test TC.NoXChange in flags
        @test TC.FailedLinesearch in flags
        @test !Optim.converged(res)

        # The trial iterate with a NaN gradient is rejected, so the returned point is finite
        g!(G, x) = (G[1] = x[1] > 0.5 ? NaN : 2 * (x[1] - 1); G)
        res = optimize(x -> (x[1] - 1)^2, g!, [0.0], BFGS(; linesearch = Optim.LineSearches.BackTracking()), Optim.Options(; show_warnings = false))
        flags = Optim.termination_flags(res)
        @test TC.FailedLinesearch in flags
        @test !(TC.GradientNotFinite in flags)
        @test !Optim.converged(res)
    end

    @testset "small trust region radius" begin
        # Steps at the kink are rejected until the radius falls below delta_min
        f(x) = abs(x[1] - 1) + (x[2] - 2)^2
        res = optimize(f, [3.0, 0.0], NewtonTrustRegion(; delta_min = 1e-8), Optim.Options(; show_warnings = false); autodiff = AutoForwardDiff())
        @test TC.SmallTrustRegionRadius in Optim.termination_flags(res)
        @test !Optim.converged(res)
    end

    @testset "constrained solvers" begin
        lower, upper = [-2.0, -2.0], [2.0, 2.0]
        res = optimize(rosen, lower, upper, x0, Fminbox(BFGS()); autodiff = AutoForwardDiff())
        @test TC.GradientNorm in Optim.termination_flags(res)
        @test Optim.converged(res)

        df = TwiceDifferentiable(rosen, x0; autodiff = AutoForwardDiff())
        res = optimize(df, TwiceDifferentiableConstraints(lower, upper), x0, IPNewton())
        @test TC.GradientNorm in Optim.termination_flags(res)
        @test Optim.converged(res)
    end

    @testset "SAMIN" begin
        res = optimize(rosen, [-2.0, -2.0], [2.0, 2.0], x0, SAMIN(; verbosity = 0), Optim.Options(; iterations = 100))
        @test collect(Optim.termination_flags(res)) == [TC.Iterations]
    end
end
