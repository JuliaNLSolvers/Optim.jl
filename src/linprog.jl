
using LinprogSolverInterface

if Pkg.installed("Clp") != nothing
    @eval using Clp
    solver = Clp
else
    solver = nothing
end

function linprog(c, A, rowlb, rowub, lb, ub)
    if solver == nothing
        error("No LP solver installed. Please run Pkg.add(\"Clp\")")
    end
    m = solver.model()
    loadproblem(m, A, lb, ub, c, rowlb, rowub)
    # LinprogSolverInterface.optimize is masked by Optim's optimize
    # maybe there's a better way
    LinprogSolverInterface.optimize(m)
    return getsolution(m)
end



