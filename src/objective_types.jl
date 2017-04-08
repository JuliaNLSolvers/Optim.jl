import NLSolversBase.OnceDifferentiable
import NLSolversBase.TwiceDifferentiable

function OnceDifferentiable{T}(f, x_seed::Vector{T}; autodiff = :finite)
    n_x = length(x_seed)
    f_calls = [1]
    g_calls = [1]
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
        gcfg = ForwardDiff.GradientConfig(x_seed)
        g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)

        fg! = (out, x) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end
    elseif autodiff == :reverse
        gcfg = ReverseDiff.GradientConfig(x_seed)
        g! = (out, x) -> ReverseDiff.gradient!(out, f, x, gcfg)

        fg! = (out, x) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ReverseDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end
    else
        error("The autodiff value $autodiff is not supported. Use :finite, :forward or :reverse.")
    end
    g = similar(x_seed)
    g!(g, x_seed)
    return OnceDifferentiable(f, g!, fg!, f(x_seed), g, copy(x_seed), copy(x_seed), f_calls, g_calls)
end

function TwiceDifferentiable{T}(f, x_seed::Vector{T}; autodiff = :finite)
    n_x = length(x_seed)
    f_calls = [1]
    g_calls = [1]
    h_calls = [1]
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
        # TODO: basically same code in :forward and :reverse
        gcfg = ForwardDiff.GradientConfig(x_seed)
        g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)

        fg! = (out, x) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end

        hcfg = ForwardDiff.HessianConfig(x_seed)
        h! = (out, x) -> ForwardDiff.hessian!(out, f, x, hcfg)
    elseif autodiff == :reverse
        gcfg = ReverseDiff.GradientConfig(x_seed)
        g! = (out, x) -> ReverseDiff.gradient!(out, f, x, gcfg)

        fg! = (out, x) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ReverseDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end
        hcfg = ReverseDiff.HessianConfig(x_seed)
        h! = (out, x) -> ReverseDiff.hessian!(out, f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite, :forward or :reverse.")
    end
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)
    g!(g, x_seed)
    h!(H, x_seed)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                               g, H, copy(x_seed),
                               copy(x_seed), copy(x_seed), f_calls, g_calls, h_calls)
end

function TwiceDifferentiable{T}(f, g!, x_seed::Array{T}; autodiff = :finite)
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
    elseif autodiff == :reverse
        hcfg = ReverseDiff.HessianConfig(x_seed)
        h! = (out, x) -> ReverseDiff.hessian!(out, f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite, :forward or :reverse.")
    end
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)
    g!(g, x_seed)
    h!(H, x_seed)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                               g, H, copy(x_seed),
                               copy(x_seed), copy(x_seed), f_calls, [1], [1])
end
#=
function TwiceDifferentiable{T}(f, g!, fg!, x_seed::Array{T}; autodiff = :finite)
    n_x = length(x_seed)
    f_calls = [1]
    if autodiff == :finite
        function h!(storage::Matrix, x::Vector)
            Calculus.finite_difference_hessian!(f, x, storage)
            return
        end
    elseif autodiff == :forward
        hcfg = ForwardDiff.HessianConfig(similar(gradient(d)))
        h! = (out, x) -> ForwardDiff.hessian!(out, f, x, hcfg)
    elseif autodiff == :reverse
        hcfg = ReverseDiff.HessianConfig(similar(gradient(d)))
        h! = (out, x) -> ReverseDiff.hessian!(out, d.f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite, :forward or :reverse.")
    g = similar(x_seed)
    g!(g, x_seed)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g, Array{T}(n_x, n_x), copy(x_seed), f_calls, [1], [0])
end
=#
function TwiceDifferentiable(d::OnceDifferentiable; autodiff = :finite)
    n_x = length(d.last_x_f)
    T = eltype(d.last_x_f)
    if autodiff == :finite
        function h!(storage::Matrix, x::Vector)
            Calculus.finite_difference_hessian!(d.f, x, storage)
            return
        end
    elseif autodiff == :forward
        hcfg = ForwardDiff.HessianConfig(similar(gradient(d)))
        h! = (out, x) -> ForwardDiff.hessian!(out, d.f, x, hcfg)
    elseif autodiff == :reverse
        hcfg = ReverseDiff.HessianConfig(similar(gradient(d)))
        h! = (out, x) -> ReverseDiff.hessian!(out, d.f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite, :forward or :reverse.")
    end
    H = Array{T}(n_x, n_x)
    h!(H, d.last_x_g)
    return TwiceDifferentiable(d.f, d.g!, d.fg!, h!, d.f_x,
                               gradient(d), H, d.last_x_f,
                               d.last_x_g, copy(d.last_x_g), d.f_calls, d.g_calls, [1])
end
