using Optim, Test
import SparseArrays: sparse
import LinearAlgebra: Tridiagonal

let
  # Why do the tests fail when this function is defined inside the @testset?
  """
  Create TwiceDifferentiable objective and TwiceDifferentiableConstraints
  representing

  min x'A*x s.t. x[i]^2 + x[i+1]^2 = 1 for i=1:2:(n-1)

  """
  function problem(A::AbstractMatrix)
    n = size(A,1)
    (mod(n,2) == 0) || error("size(A,1) must be even")
    ncon = n ÷ 2
    h0 = A
    g0 = zeros(eltype(A),n)
    x0 = zeros(n)
    function grad!(g,x)
      for i in 1:n
        g[i] = 2*x'*A[:,i]
      end
    end
    function hess!(h,x)
      copyto!(h,A)
    end
    function obj(x)
      return(x'*A*x)
    end
    f = Optim.TwiceDifferentiable(obj, grad!,
                                  hess!, x0, zero(eltype(A)), g0, h0 )
    function con_c!(c,x)
      for i in 1:ncon
        c[i] = x[2*i-1]^2 + x[2*i]^2
      end
      c
    end
    Jstore = similar(A,n÷2,n)
    Jstore .= zero(eltype(A))
    function con_j!(J,x)
      for i in 1:ncon
        J[i,2*i-1] = 2x[2*i-1]
        J[i,2*i] = 2x[2*i]
      end
    end
    function con_hl!(h,x,λ)
      for i in 1:n
        h[i,i] += 2λ[(i+1)÷2]
      end
    end
    lx = fill(-Inf,n)
    ux = fill(Inf,n)
    lc = ones(n÷2)
    uc = ones(n÷2)
    fc = Optim.TwiceDifferentiableConstraints(con_c!, con_j!, con_hl!, lx, ux, lc, uc)
    return(f, fc, Jstore)
  end


  @testset "hessian and jacobian alternate types" begin
    n = 8
    A = zeros(n,n)
    x0 = ones(n)
    for i in 1:n
      A[i,i] = 2.0
      if (i>1)
        A[i-1,i] = 1.0
        A[i,i-1] = 1.0
      end
    end

    f,con,J = problem(A)
    dense_sol = Optim.optimize(f,con,x0,Optim.IPNewton(conJstorage=J))

    @testset "sparse" begin
      f,con,J = problem(sparse(A))
      sparse_sol = Optim.optimize(f,con,x0,Optim.IPNewton(conJstorage=J))
      @test dense_sol.minimum ≈ sparse_sol.minimum
      @test dense_sol.minimizer ≈ sparse_sol.minimizer
    end

    @testset "tridiagonal" begin
      f,con,J = problem(Tridiagonal(A))
      tridiagonal_sol = Optim.optimize(f,con,x0,Optim.IPNewton(conJstorage=J))
      @test dense_sol.minimum ≈ tridiagonal_sol.minimum
      @test dense_sol.minimizer ≈ tridiagonal_sol.minimizer
    end
  end

end
