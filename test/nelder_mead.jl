using Optim

# Test Optim.nelder_mead for all functions except Large Polynomials in Optim.UnconstrainedProblems.examples
for (name, prob) in Optim.UnconstrainedProblems.examples
    f_prob = prob.f
    if name == "Large Polynomial"
        res = Optim.nelder_mead(f_prob, prob.initial_x, iterations=1_000_000)
    else
        res = Optim.nelder_mead(f_prob, prob.initial_x)
    end
    @assert norm(res.minimum - prob.solutions) < 1e-2
end

function f(x::Vector)
  (100.0 - x[1])^2 + x[2]^2
end

function rosenbrock(x::Vector)
  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

initial_x = [0.0, 0.0]

results = Optim.nelder_mead(f, initial_x)

@assert results.f_converged
@assert norm(results.minimum - [100.0, 0.0]) < 0.01
@assert length(results.trace.states) == 0

results = Optim.nelder_mead(rosenbrock, initial_x)

@assert results.f_converged
@assert norm(results.minimum - [1.0, 1.0]) < 0.01
@assert length(results.trace.states) == 0


