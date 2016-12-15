
import LineSearches

let
   methods = [ LBFGS(), ConjugateGradient(),
               LBFGS(extrapolate=true, linesearch = LineSearches.interpbacktrack!) ]
   msgs = ["LBFGS Default Options: ",  "CG Default Options: ",
          "LBFGS + Backtracking + Extrapolation: " ]

   println("--------------------")
   println("Rosenbrock Example: ")
   println("--------------------")
   rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
   for (method, msg) in zip(methods, msgs)
      results = Optim.optimize(rosenbrock, zeros(2), method=method)
      println(msg, "g_calls = ", results.g_calls, ", f_calls = ", results.f_calls)
   end

   println("--------------------------------------")
   println("p-Laplacian Example (preconditioned): ")
   println("--------------------------------------")
   plap(U; n=length(U)) = (n-1) * sum( (0.1 + diff(U).^2).^2 ) - sum(U) / (n-1)
   plap1(U; n=length(U), dU = diff(U), dW = 4 * (0.1 + dU.^2) .* dU) =
                           (n-1) * ([0.0; dW] - [dW; 0.0]) - ones(U) / (n-1)
   precond(x::Vector) = precond(length(x))
   precond(n::Number) = spdiagm( ( -ones(n-1), 2*ones(n), -ones(n-1) ),
                                 (-1,0,1), n, n) * (n+1)
   df = DifferentiableFunction( X->plap([0;X;0]),
                                (X, G)->copy!(G, (plap1([0;X;0]))[2:end-1]) )
   GRTOL = 1e-6
   N = 100
   initial_x = zeros(N)
   P = precond(initial_x)
   methods = [ LBFGS(P=P), ConjugateGradient(P=P),
         LBFGS(extrapolate=true, linesearch = LineSearches.interpbacktrack!, P=P) ]

   for (method, msg) in zip(methods, msgs)
      results = Optim.optimize(df, copy(initial_x), method=method)
      println(msg, "g_calls = ", results.g_calls, ", f_calls = ", results.f_calls)
   end
end
