module TestOptim

using Test
import Optim
import MathOptInterface as MOI

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_SolverName()
    @test MOI.get(Optim.Optimizer(), MOI.SolverName()) == "Optim"
end

function test_supports_incremental_interface()
    @test MOI.supports_incremental_interface(Optim.Optimizer())
end

function test_MOI_Test()
    model = MOI.instantiate(Optim.Optimizer, with_cache_type = Float64)
    MOI.set(model, MOI.Silent(), true)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            atol = 1e-6,
            rtol = 1e-6,
            optimal_status = MOI.LOCALLY_SOLVED,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.ConstraintName,
                MOI.VariableName,
                MOI.ObjectiveBound,
                MOI.DualObjectiveValue,
                MOI.SolverVersion,
                MOI.ConstraintDual,
            ],
        ),
        exclude = [
            # FIXME Incorrect solution
            r"test_nonlinear_expression_hs071$",
            # FIXME Starting value is not feasible
            # See https://github.com/JuliaNLSolvers/Optim.jl/issues/1071
            r"test_nonlinear_expression_hs071_epigraph$",
            # FIXME objective off by 1, seems fishy
            r"test_objective_FEASIBILITY_SENSE_clears_objective$",
            # No objective
            r"test_attribute_SolveTimeSec$",
            r"test_attribute_RawStatusString$",
            # Detecting infeasibility not supported
            r"test_solve_TerminationStatus_DUAL_INFEASIBLE$",
        ],
    )
    return
end

end # module TestOptim

TestOptim.runtests()
