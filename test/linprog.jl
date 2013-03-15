# min -x
# s.t. 2x + y <= 1.5
# x,y >= 0
# solution is (0.75,0) with objval -0.75

sol = linprog([-1,0],[2 1],'<',1.5)
@assert sol.status == :Optimal && norm(sol.objval+0.75) < 1e-7 && norm(sol.sol - [0.75,0.0]) < 1e-7
linprog([-1,0],sparse([2 1]),'<',1.5)
@assert sol.status == :Optimal && norm(sol.objval+0.75) < 1e-7 && norm(sol.sol - [0.75,0.0]) < 1e-7

# test infeasible problem:
# min x
# s.t. 2x+y <= -1
# x,y >= 0
sol = linprog([1,0],[2 1],'<',-1)
@assert sol.status == :Infeasible

# test unbounded problem:
# min -x-y
# s.t. -x+2y <= 0
# x,y >= 0
sol = linprog([-1,-1],[-1 2],'<',[0])
@assert sol.status == :Unbounded

