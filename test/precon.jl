
using Optim


# this implements the 1D p-laplacian (p = 4)
#   F(u) = ∑_{i=1}^{N} h (W(u_i') - ∑_{i=1}^{N-1} h u_i
#  where u_i' = (u_i - u_{i-1})/h
#  plap: implements the functional without boundary condition
#  objective : applies Dirichlet boundary conditions u_0 = u_N = 0
W(r) = (1+r.^2).^2
W1(r) = 4 * (1+r.^2) .* r
h(U) = length(U)-1
plap(U) = h(U) * sum( W(diff(U)) ) - sum(U) / h(U)
plap1(U) = h(U) * ([0.0; W1(diff(U))] - [W1(diff(U)); 0.0]) - ones(U) / h(U)
plap_objective(X::Vector) = plap([0;X;0])
plap_objective_gradient!(X::Vector, G) = copy!(G, plap1([0;X;0])[2:end-1])

# discrete laplacian as a preconditioner
# note for this example we could just use the hessian itself, but
#  this is not the point here, we just want to check that preconditioning
#  works fine.
lap_precond(X::Vector) =
    spdiagm( ( (-1) * ones(length(X)-1),
                 2  * ones(length(X)),
               (-1) * ones(length(X)-1) ),
             (-1,0,1), length(X), length(X) ) * (length(X)+1)


df = DifferentiableFunction(plap_objective, plap_objective_gradient!)
GRTOL = 1e-8

println("Test a basic preconditioning example")
for N in (10, 100, 1000)
    println("N = ", N)
    x0 = zeros(N)
    Plap = lap_precond(x0)
    ID = nothing
    for Optimiser in (GradientDescent, ConjugateGradient, LBFGS)
        for (P, wwo) in zip((ID, Plap), (" WITHOUT", " WITH"))
            results = Optim.optimize(df, copy(x0),
                                     method=Optimiser(P = P),
                                     ftol = 1e-32, grtol = GRTOL )
            println(Optimiser, wwo, 
                    " preconditioning : g_calls = ", results.g_calls,
                    ", f_calls = ", results.f_calls)
            if (Optimiser == GradientDescent) && (N > 15) && (P == ID)
                println("    (gradient descent is not expected to converge)")
            else
                @assert Optim.converged(results)
            end
        end
    end
end


