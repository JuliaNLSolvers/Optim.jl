using Optim
using Test
import Aqua
import ExplicitImports
import JET

@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(Optim)
    end

    @testset "ExplicitImports" begin
        # No implicit imports (`using XY`)
        @test ExplicitImports.check_no_implicit_imports(
            Optim;
            # ExplicitImports does not support `@enumx`
            # Ref https://github.com/JuliaTesting/ExplicitImports.jl/issues/73
            allow_unanalyzable = (Optim.TerminationCode,),
        ) === nothing

        # All explicit imports (`using XY: Z`) are loaded via their owners
        @test ExplicitImports.check_all_explicit_imports_via_owners(
            Optim;
            ignore = (
                # ExplicitImports does currently not ignore non-public names of main package in extensions
                # Ref https://github.com/JuliaTesting/ExplicitImports.jl/issues/92
                :LinearAlgebra,
            ),
        ) === nothing

        # No explicit imports (`using XY: Z`) that are not used
        @test ExplicitImports.check_no_stale_explicit_imports(
            Optim;
            # ExplicitImports does not support `@enumx`
            # Ref https://github.com/JuliaTesting/ExplicitImports.jl/issues/73
            allow_unanalyzable = (Optim.TerminationCode,),
        ) === nothing

        # Nothing is accessed via modules other than its owner
        @test ExplicitImports.check_all_qualified_accesses_via_owners(Optim) === nothing

        # Optim currently accesses many non-public names
        @test_broken ExplicitImports.check_all_qualified_accesses_are_public(Optim) === nothing

        # No self-qualified accesses
        @test ExplicitImports.check_no_self_qualified_accesses(Optim) === nothing
    end

    @testset "JET" begin
        # Check that there are no undefined global references and undefined field accesses
        res = JET.report_package(Optim; target_defined_modules = true, mode = :typo, toplevel_logger = nothing)
        reports = JET.get_reports(res)
        @test_broken isempty(reports)
        @test length(reports) <= 36


        # Analyze methods based on their declared signature
        res = JET.report_package(Optim; target_defined_modules = true, toplevel_logger = nothing)
        reports = JET.get_reports(res)
        @test_broken isempty(reports)
        @test length(reports) <= 12
    end
end