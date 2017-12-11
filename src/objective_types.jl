NonDifferentiable(f, g!,     x_seed::AbstractArray) = NonDifferentiable(f, x_seed)
NonDifferentiable(f, g!, h!, x_seed::AbstractArray) = NonDifferentiable(f, x_seed)

function OnceDifferentiable(f, F::T, x_seed::AbstractArray{T}; autodiff = :finite) where T <: Real

    n_x = length(x_seed)
    if autodiff == :finite
        function g!(storage, x)
            Calculus.finite_difference!(f, x, storage, :central)
            return
        end
        function fg!(storage, x)
            g!(storage, x)
            return f(x)
        end
    elseif autodiff == :forward
        gcfg = ForwardDiff.GradientConfig(f, x_seed)
        g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)

        fg! = (out, x) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end
    # elseif autodiff == :reverse
    #     gcfg = ReverseDiff.GradientConfig(x_seed)
    #     g! = (out, x) -> ReverseDiff.gradient!(out, f, x, gcfg)
    #
    #     fg! = (out, x) -> begin
    #         gr_res = DiffBase.DiffResult(zero(T), out)
    #         ReverseDiff.gradient!(gr_res, f, x, gcfg)
    #         DiffBase.value(gr_res)
    #     end
    else
        error("The autodiff value $autodiff is not support. Use :finite or :forward.")
    end
    g = similar(x_seed)
    return OnceDifferentiable(f, g!, fg!, zero(T), g, copy(x_seed))
end

function TwiceDifferentiable(f, g!, F::Real, x_seed::AbstractVector{T}; autodiff = :finite) where T
    n_x = length(x_seed)
    f_calls = [1]
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    if autodiff == :finite
        function h!(storage, x)
            Calculus.finite_difference_hessian!(f, x, storage)
            return
        end
    elseif autodiff == :forward
        hcfg = ForwardDiff.HessianConfig(similar(x_seed))
        h! = (out, x) -> ForwardDiff.hessian!(out, f, x, hcfg)
    # elseif autodiff == :reverse
    #     hcfg = ReverseDiff.HessianConfig(x_seed)
    #     h! = (out, x) -> ReverseDiff.hessian!(out, f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite or :forward.")
    end
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)
    g!(g, x_seed)
    h!(H, x_seed)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                               g, H, copy(x_seed),
                               copy(x_seed), copy(x_seed), f_calls, [1], [1])
end

TwiceDifferentiable(d::NonDifferentiable, F::T, x_seed::AbstractVector{T} = d.x_f; autodiff = :finite) where {T<:Real} =
TwiceDifferentiable(d.f, F, x_seed; autodiff = autodiff)
function TwiceDifferentiable(d::OnceDifferentiable, F::T, x_seed::AbstractVector{T} = d.x_f; autodiff = :finite) where T<:Real
    n_x = length(x_seed)
    if autodiff == :finite
        function h!(storage, x)
            Calculus.finite_difference_hessian!(d.f, x, storage)
            return
        end
    elseif autodiff == :forward
        hcfg = ForwardDiff.HessianConfig(similar(gradient(d)))
        h! = (out, x) -> ForwardDiff.hessian!(out, d.f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite or :forward.")
    end
    H = Array{T}(n_x, n_x)
    return TwiceDifferentiable(d.f, d.df, d.fdf, h!, F, copy(gradient(d)), H, similar(x_seed))
end
    
# Automatically create the fg! helper function if only f, g! and h! is provided
function TwiceDifferentiable(f, F::T, x::AbstractVector{T}; autodiff = :finite) where T<:Real
    if autodiff == :finite
        function g!(storage::Vector, x::Vector)
            Calculus.finite_difference!(f, x, storage, :central)
            return
        end
        function fg!(storage::Vector, x::Vector)
            g!(storage, x)
            return f(x)
        end
        function h!(storage::Matrix, x::Vector)
            Calculus.finite_difference_hessian!(f, x, storage)
            return
        end
    elseif autodiff == :forward
        gcfg = ForwardDiff.GradientConfig(f, x)
        g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)

        fg! = (out, x) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end

        hcfg = ForwardDiff.HessianConfig(f, x)
        h! = (out, x) -> ForwardDiff.hessian!(out, f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite or :forward.")
    end
    TwiceDifferentiable(f, g!, fg!, h!, F, x)
    n = length(x)
    H = similar(x, n, n)
    return TwiceDifferentiable(f, g!, fg!, h!, zero(T),
                               similar(x), H, similar(x),
                               similar(x), similar(x), [0,], [0,], [0,])
end
