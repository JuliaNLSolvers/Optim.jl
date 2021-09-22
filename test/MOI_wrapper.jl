module TestOptim

import Optim
using MathOptInterface
using Test

const MOI = MathOptInterface

const OPTIMIZER_CONSTRUCTOR = MOI.OptimizerWithAttributes(
    Optim.Optimizer{Float64},
    MOI.Silent() => true,
)
const OPTIMIZER = MOI.instantiate(OPTIMIZER_CONSTRUCTOR)
const CACHED = MOI.Utilities.CachingOptimizer(
    MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
    OPTIMIZER,
)

const CONFIG = MOI.DeprecatedTest.Config(
    atol = 1e-6,
    rtol = 1e-6,
    duals = false,
    query = false,
    infeas_certificates = false,
    optimal_status = MOI.LOCALLY_SOLVED,
)

function test_SolverName()
    @test MOI.get(OPTIMIZER, MOI.SolverName()) == "Optim"
end

function test_supports_incremental_interface()
    @test MOI.supports_incremental_interface(OPTIMIZER)
end

function test_nlp()
    MOI.DeprecatedTest.nlptest(CACHED, CONFIG, String[
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
    ])
end

# This function runs all functions in this module starting with `test_`.
function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

end # module TestOptim

TestOptim.runtests()
