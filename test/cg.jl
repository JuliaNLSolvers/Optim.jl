using Optim

for (name, prob) in UnconstrainedProblems.examples
    if prob.isdifferentiable
        if name == "Himmelbrau"
            continue
        end
        df = DifferentiableFunction(prob.f, prob.g!)
        res = Optim.cg(df, prob.initial_x)
        if length(prob.solutions) == 1
            @assert norm(res.minimum - prob.solutions[1]) < 1e-2
        end
    end
end

for (name, prob) in ConstrainedProblems.pexamples
    if prob.isdifferentiable
        df = DifferentiableFunction(prob.f, prob.g!)
        res = Optim.cg(df, prob.initial_x, constraints=prob.constraints)
        if length(prob.solutions) == 1
            @assert norm(res.minimum - prob.solutions[1]) < 1e-2
        end
    end
end


# Functions of arrays
let
objective(X, B) = sum((X.-B).^2)/2

function objective_gradient!(X, G, B)
    for i = 1:length(G)
        G[i] = X[i]-B[i]
    end
end

srand(1)
B = rand(2,2)
df = Optim.DifferentiableFunction(X -> objective(X, B), (X, G) -> objective_gradient!(X, G, B))
results = Optim.cg(df, rand(2,2))
@assert Optim.converged(results)
@assert results.f_minimum < 1e-8
end
