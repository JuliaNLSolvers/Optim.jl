module TestOptim

using Test
import Optim
import MathOptInterface
const MOI = MathOptInterface

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
    @test MOI.get(Optim.moi_optimizer(), MOI.SolverName()) == "Optim"
end

function test_supports_incremental_interface()
    @test MOI.supports_incremental_interface(Optim.moi_optimizer())
end

function test_MOI_Test()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        Optim.moi_optimizer(),
    )
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
        exclude = String[
            # No objective
            "test_attribute_SolveTimeSec",
            "test_attribute_RawStatusString",
            # FIXME The hessian callback for constraints is called with
            # `λ = [-Inf, 0.0]` and then we get `NaN`, ...
            "expression_hs071",
            # Terminates with `OTHER_ERROR`
            "test_objective_ObjectiveFunction_duplicate_terms",
            "test_objective_ObjectiveFunction_constant",
            "test_objective_ObjectiveFunction_VariableIndex",
            "test_objective_FEASIBILITY_SENSE_clears_objective",
            "test_nonlinear_expression_hs109",
            "test_objective_qp_ObjectiveFunction_zero_ofdiag",
            "test_objective_qp_ObjectiveFunction_edge_cases",
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            "test_solve_result_index",
            "test_modification_transform_singlevariable_lessthan",
            "test_modification_delete_variables_in_a_batch",
            "test_modification_delete_variable_with_single_variable_obj",
        ],
    )
    return
end

end # module TestOptim

TestOptim.runtests()
