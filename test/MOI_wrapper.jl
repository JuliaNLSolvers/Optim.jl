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
    @test MOI.get(Optim.Optimizer(), MOI.SolverName()) == "Optim"
end

function test_supports_incremental_interface()
    @test MOI.supports_incremental_interface(Optim.Optimizer())
end

function test_MOI_Test()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        Optim.Optimizer(),
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            atol = 1e-6,
            rtol = 1e-6,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.ConstraintName,
                MOI.VariableName,
                MOI.ObjectiveBound,
                MOI.DualObjectiveValue,
            ],
        ),
        exclude = String[
            # No objective
            "test_attribute_SolveTimeSec",
            "test_attribute_RawStatusString",
            "test_nonlinear_without_objective",
            # FIXME INVALID_MODEL should be returned
            "test_nonlinear_invalid",
            # FIXME The hessian callback for constraints is called with
            # `λ = [-Inf, 0.0]` and then we get `NaN`, ...
            "hs071",
            # There are nonlinear constraints so we need `IPNewton` but `IPNewton` needs a hessian.
            "test_nonlinear_hs071_no_hessian",
            # FIXME Here there is no hessian but there is a hessian-vector product, can `IPNewton` work with that ?
            "test_nonlinear_hs071_hessian_vector_product",
            # FIXME needs https://github.com/jump-dev/MathOptInterface.jl/pull/1625
            "test_nonlinear_hs071_NLPBlockDual",
            #  - CachingOptimizer does not throw if optimizer not attached
            "test_model_copy_to_UnsupportedAttribute",
            "test_model_copy_to_UnsupportedConstraint",
        ],
    )
    return
end

function test_nlp()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        Optim.Optimizer(),
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.DeprecatedTest.nlptest(
        model,
        MOI.DeprecatedTest.Config(
            atol = 1e-6,
            rtol = 1e-6,
            duals = false,
            query = false,
            infeas_certificates = false,
            optimal_status = MOI.LOCALLY_SOLVED,
        ),
        String[
            # FIXME The hessian callback for constraints is called with
            # `λ = [-Inf, 0.0]` and then we get `NaN`, ...
            "hs071",
            # There are nonlinear constraints so we need `IPNewton` but `IPNewton` needs a hessian.
            "hs071_no_hessian", "feasibility_sense_with_objective_and_no_hessian",
            # FIXME Here there is no hessian but there is a hessian-vector product, can `IPNewton` work with that ?
            "hs071_hessian_vector_product_test",
            # No objective, would be fixed by https://github.com/jump-dev/MathOptInterface.jl/issues/1397
            "feasibility_sense_with_no_objective_and_with_hessian",
            "feasibility_sense_with_no_objective_and_no_hessian",
            # Affine objective, would be fixed by https://github.com/jump-dev/MathOptInterface.jl/issues/1397
            "nlp_objective_and_moi_objective",
            #"feasibility_sense_with_no_objective_and_with_hessian",
        ],
    )
end

end # module TestOptim

TestOptim.runtests()
