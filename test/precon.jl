
# this implements the 1D p-laplacian (p = 4)
#      F(u) = ∑_{i=1}^{N} h (W(u_i') - ∑_{i=1}^{N-1} h u_i
#  where u_i' = (u_i - u_{i-1})/h
#  plap: implements the functional without boundary condition
#  preconditioner is a discrete laplacian, which defines a metric
#     equivalent (in the limit h → 0) to that induced by the hessian, but
#     does not approximate the hessian explicitly.

using Optim
plap(U; n=length(U)) = (n-1) * sum( (0.1 + diff(U).^2).^2 ) - sum(U) / (n-1)
plap1(U; n=length(U), dU = diff(U), dW = 4 * (0.1 + dU.^2) .* dU) =
                        (n-1) * ([0.0; dW] - [dW; 0.0]) - ones(U) / (n-1)
precond(x::Vector) = precond(length(x))
precond(n::Number) = spdiagm( ( -ones(n-1), 2*ones(n), -ones(n-1) ),
                              (-1,0,1), n, n) * (n+1)
df = DifferentiableFunction( X->plap([0;X;0]),
                             (X, G)->copy!(G, (plap1([0;X;0]))[2:end-1]) )

GRTOL = 1e-6

println("Test a basic preconditioning example")
for N in (10, 50, 250)
    println("N = ", N)
    x0 = zeros(N)
    Plap = precond(x0)
    ID = nothing
    for Optimiser in (GradientDescent, ConjugateGradient, LBFGS)
        for (P, wwo) in zip((ID, Plap), (" WITHOUT", " WITH"))
            results = Optim.optimize(df, copy(x0),
                                     method=Optimiser(P = P),
                                     f_tol = 1e-32, g_tol = GRTOL )
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
