@testset "Newton" begin

    function f_1(x::Vector)
        (x[1] - 5.0)^4
    end

    function g!_1(storage::Vector, x::Vector)
        storage[1] = 4.0 * (x[1] - 5.0)^3
    end

    function h!_1(storage::Matrix, x::Vector)
        storage[1, 1] = 12.0 * (x[1] - 5.0)^2
    end

    initial_x = [0.0]

    Optim.optimize(NonDifferentiable(f_1, initial_x), [0.0], Newton())
    Optim.optimize(OnceDifferentiable(f_1, g!_1, initial_x), [0.0], Newton())

    options = Optim.Options(store_trace = false, show_trace = false, extended_trace = true)
    results = Optim.optimize(f_1, g!_1, h!_1, [0.0], Newton(), options)
    @test_throws ErrorException Optim.x_trace(results)
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [5.0]) < 0.01
    
    eta = 0.9

    function f_2(x::Vector)
        (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
    end

    function g!_2(storage::Vector, x::Vector)
        storage[1] = x[1]
        storage[2] = eta * x[2]
    end

    function h!_2(storage::Matrix, x::Vector)
        storage[1, 1] = 1.0
        storage[1, 2] = 0.0
        storage[2, 1] = 0.0
        storage[2, 2] = eta
    end

    results = Optim.optimize(f_2, g!_2, h!_2, [127.0, 921.0], Newton())
    @test_throws ErrorException Optim.x_trace(results)
    @test Optim.g_converged(results)
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01
    test_summary(results, "Newton's Method")

    @testset "newton in concave region" begin
        prob = MultivariateProblems.UnconstrainedProblems.examples["Himmelblau"]
        res = optimize(
            MVP.objective(prob),
            MVP.gradient(prob),
            MVP.hessian(prob),
            [0.0, 0.0],
            Newton(),
        )
        @test norm(Optim.minimizer(res) - prob.solutions) < 1e-9
    end
    
    @testset "Optim problems" begin
        run_optim_tests(Newton(); skip = ("Trigonometric",), show_name = debug_printing)
    end

    wrap_solver(Hg_solver) = (d, state, method) -> begin
        H = NLSolversBase.hessian(d)
        g = NLSolversBase.gradient(d)
        state.s .= Hg_solver(H, g)
        nothing
    end
    
    @testset "Custom Solvers" begin

        # Custom solver using LU decomposition
        custom_solve(H, g) = -(lu(H) \ g)

        result = optimize(f_2, g!_2, h!_2, [10.0, 20.0], Newton(solve=wrap_solver(custom_solve)))
        @test Optim.g_converged(result)
        @test norm(Optim.minimizer(result) - [0.0, 0.0]) < 0.01
        
        # Custom solver using QR decomposition
        qr_solve(H, g) = -(qr(H) \ g)
        result2 = optimize(f_2, g!_2, h!_2, [5.0, 5.0], Newton(solve=wrap_solver(qr_solve)))
        @test Optim.g_converged(result2)
        @test norm(Optim.minimizer(result2) - [0.0, 0.0]) < 0.01

        # Simple solver
        simple_solve(H, g) = -(H \ g)
        result3 = optimize(f_2, g!_2, h!_2, [3.0, 4.0], Newton(solve=wrap_solver(simple_solve)))
        @test Optim.g_converged(result3)
        @test norm(Optim.minimizer(result3) - [0.0, 0.0]) < 0.01

        # Solver with specialized matrx types--can add more late i.e., BlockBanded, Block, etc. for right now, only testing StaticArrays
        using StaticArrays
        static_solve(H, g) = -(SMatrix{2, 2}(H) \ SVector{2}(g))
        result_static = optimize(f_2, g!_2, h!_2, [6.0, 7.0], Newton(solve=wrap_solver(static_solve)))
        @test Optim.g_converged(result_static)
        @test norm(Optim.minimizer(result_static) - [0.0, 0.0]) < 0.01
    end

    @testset "BlockArray Hessian with custom solver" begin
        using BlockArrays

        function f(x)
            return sum((x .- 1.0).^2)
        end

        function g!(storage, x)
            storage .= 2.0 .* (x .- 1.0)
        end

        function h!(H::BlockArray{<:Any,2}, x)
            for i in 1:size(H.blocks, 1)
                blk = H.blocks[i, i]
                fill!(blk, 0.0)
                for j in 1:min(size(blk)...)
                    blk[j,j] = 2.0
                end
            end
            return nothing
        end

        # --- Setup ---
        block_sizes = (2, 3)
        n = sum(block_sizes)

        x0 = zeros(n)
        initial_f = f(x0)
        initial_g = similar(x0)
        g!(initial_g, x0)

        H_data = zeros(n, n)
        H_block = BlockArray(H_data, [2, 3], [2, 3])
        h!(H_block, x0)

        custom_solve(H::BlockArray, g) = -(Matrix(H) \ g)

        td = TwiceDifferentiable(f, g!, h!, x0, initial_f, initial_g, H_block)
        result = optimize(td, x0, Newton(solve=wrap_solver(custom_solve)))

        @test Optim.g_converged(result)
        @test typeof(NLSolversBase.hessian(td)) <: BlockArray
        @test typeof(result.minimizer) == Vector{Float64}
        @test norm(result.minimizer .- ones(n)) < 1e-6
    end

    @testset "Hessian Types" begin
        using SparseArrays
        
        # Test sparse solver
        sparse_solve(H, g) = -(sparse(H) \ g)
        result_sparse = optimize(f_2, g!_2, h!_2, [5.0, 5.0], Newton(solve=wrap_solver(sparse_solve)))
        @test Optim.g_converged(result_sparse)
        @test norm(Optim.minimizer(result_sparse) - [0.0, 0.0]) < 0.01
        
        # Test default solver handles both dense and sparse correctly
        using LinearAlgebra
        result_default = optimize(f_2, g!_2, h!_2, [3.0, 4.0], Newton())
        @test Optim.g_converged(result_default)
        @test norm(Optim.minimizer(result_default) - [0.0, 0.0]) < 0.01
    end

    @testset "Block Tridiagonal System - Kalman Smoothing" begin
        using LinearAlgebra, Random, Optim

        # ----------------------------
        # Utility: Block tridiagonal solver
        # ----------------------------
        function block_thomas_solve(H, g, block_size)
            n_blocks = length(g) ÷ block_size
            B = [Matrix(H[(i-1)*block_size+1:i*block_size, (i-1)*block_size+1:i*block_size]) for i in 1:n_blocks]
            A = [Matrix(H[i*block_size+1:(i+1)*block_size, (i-1)*block_size+1:i*block_size]) for i in 1:n_blocks-1]
            C = [Matrix(H[(i-1)*block_size+1:i*block_size, i*block_size+1:(i+1)*block_size]) for i in 1:n_blocks-1]
            d = [g[(i-1)*block_size+1:i*block_size] for i in 1:n_blocks]
            B_work = copy(B)
            d_work = copy(d)
            for i in 2:n_blocks
                L = A[i-1] / B_work[i-1]
                B_work[i] .-= L * C[i-1]
                d_work[i] .-= L * d_work[i-1]
            end
            x = Vector{Vector{Float64}}(undef, n_blocks)
            x[end] = B_work[end] \ d_work[end]
            for i in n_blocks-1:-1:1
                x[i] = B_work[i] \ (d_work[i] - C[i] * x[i+1])
            end
            return vcat(x...)
        end

        # ----------------------------
        # Problem setup
        # ----------------------------
        function create_kalman_problem(T::Int, D::Int)
            Random.seed!(1)
            A = 0.9I + 0.1 * randn(D, D)
            C = randn(2, D)
            Q = I + 0.1 * randn(D, D); Q = Q'Q
            R = I + 0.1 * randn(2, 2); R = R'R
            x0 = randn(D)
            P0 = Matrix{Float64}(I, D, D)
            y = randn(2, T)
            w = ones(T)
            return (; A, C, Q, R, x0, P0, y, w)
        end

        # ----------------------------
        # Negative log-likelihood
        # ----------------------------
        function nll(x_vec, p)
            X = reshape(x_vec, size(p.A, 1), size(p.y, 2))
            R_chol = cholesky(Symmetric(p.R)).U
            Q_chol = cholesky(Symmetric(p.Q)).U
            P0_chol = cholesky(Symmetric(p.P0)).U

            ll = sum(abs2, P0_chol \ (X[:,1] - p.x0))
            for t in 1:size(p.y, 2)
                if t > 1
                    ll += sum(abs2, Q_chol \ (X[:,t] - p.A * X[:,t-1]))
                end
                ll += p.w[t] * sum(abs2, R_chol \ (p.y[:,t] - p.C * X[:,t]))
            end
            return 0.5 * ll
        end

        # ----------------------------
        # Gradient
        # ----------------------------
        function g!(g, x_vec, p)
            D, T = size(p.A, 1), size(p.y, 2)
            X = reshape(x_vec, D, T)
            R_chol = cholesky(Symmetric(p.R))
            Q_chol = cholesky(Symmetric(p.Q))
            P0_chol = cholesky(Symmetric(p.P0))
            C_inv_R = (R_chol \ p.C)'
            A_inv_Q = (Q_chol \ p.A)'

            grad = zeros(D, T)

            grad[:,1] .= A_inv_Q * (X[:,2] - p.A * X[:,1]) +
                        p.w[1] * C_inv_R * (p.y[:,1] - p.C * X[:,1]) -
                        (P0_chol \ (X[:,1] - p.x0))

            for t in 2:T-1
                grad[:,t] .= p.w[t] * C_inv_R * (p.y[:,t] - p.C * X[:,t]) -
                            Q_chol \ (X[:,t] - p.A * X[:,t-1]) +
                            A_inv_Q * (X[:,t+1] - p.A * X[:,t])
            end

            grad[:,T] .= p.w[T] * C_inv_R * (p.y[:,T] - p.C * X[:,T]) -
                        Q_chol \ (X[:,T] - p.A * X[:,T-1])

            g .= vec(-grad)  
            return nothing
        end

        # ----------------------------
        # Hessian
        # ----------------------------
        function h!(H, x_vec, p)
            D, T = size(p.A, 1), size(p.y, 2)
            inv_R = inv(p.R)
            inv_Q = inv(p.Q)
            inv_P0 = inv(p.P0)
            yt_xt = p.C' * inv_R * p.C
            xt_xt_1 = inv_Q
            xt1_xt = p.A' * inv_Q * p.A
            x0_term = inv_P0

            diag = Vector{Matrix{Float64}}(undef, T)
            sub = fill(-inv_Q * p.A, T-1)
            sup = fill(-p.A' * inv_Q, T-1)

            diag[1] = p.w[1] * yt_xt + xt1_xt + x0_term
            for t in 2:T-1
                diag[t] = p.w[t] * yt_xt + xt_xt_1 + xt1_xt
            end
            diag[T] = p.w[T] * yt_xt + xt_xt_1

            H_mat = zeros(D*T, D*T)
            for t in 1:T
                idx = (t-1)*D + 1 : t*D
                H_mat[idx, idx] = diag[t]
                if t < T
                    H_mat[idx, idx .+ D] = sup[t]
                    H_mat[idx .+ D, idx] = sub[t]
                end
            end
            mul!(H, -1.0, H_mat)
            return nothing
        end

        # ----------------------------
        # Main test
        # ----------------------------
        function run()
            T, D = 50, 4
            p = create_kalman_problem(T, D)
            x0 = zeros(T * D)

            f = x -> nll(x, p)
            g = (storage, x) -> g!(storage, x, p)
            h = (storage, x) -> h!(storage, x, p)

            println("Standard Newton:")
            res1 = optimize(f, g, h, copy(x0), Newton())
            println("  Iterations: ", Optim.iterations(res1))

            println("Block Thomas Newton:")
            res2 = optimize(f, g, h, copy(x0), Newton(solve = wrap_solver((H, g) -> block_thomas_solve(H, g, D)), Optim.Options(show_trace=true)))
            println("  Iterations: ", Optim.iterations(res2))

            @test Optim.g_converged(res1)
            @test Optim.g_converged(res2)
            @test norm(Optim.minimizer(res1) - Optim.minimizer(res2)) < 1e-8
        end
    end
end

