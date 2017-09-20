import NLSolversBase.NonDifferentiable
import NLSolversBase.OnceDifferentiable
import NLSolversBase.TwiceDifferentiable


abstract type UninitializedObjective <: AbstractObjective end
mutable struct UninitializedNonDifferentiable <: UninitializedObjective
    f
end
# The user friendly/short form NonDifferentiable constructor
NonDifferentiable(f) = UninitializedNonDifferentiable(f)
NonDifferentiable(u::UninitializedNonDifferentiable, x::AbstractArray) = NonDifferentiable(u.f, x)

mutable struct UninitializedOnceDifferentiable{T} <: UninitializedObjective
    f
    g!
    fg!::T
end
# The user friendly/short form OnceDifferentiable constructor
OnceDifferentiable(f, g!, fg!) = UninitializedOnceDifferentiable(f, g!,      fg!)
OnceDifferentiable(f, g!)      = UninitializedOnceDifferentiable(f, g!,      nothing)
OnceDifferentiable(f)          = UninitializedOnceDifferentiable(f, nothing, nothing)
OnceDifferentiable(u::UninitializedOnceDifferentiable, x::AbstractArray) = OnceDifferentiable(u.f, u.g!, u.fg!, x)
OnceDifferentiable(u::UninitializedOnceDifferentiable{Void}, x::AbstractArray) = OnceDifferentiable(u.f, u.g!, x)

mutable struct UninitializedTwiceDifferentiable{Tf, Tfg, Th} <: UninitializedObjective
    f
    g!::Tf
    fg!::Tfg
    h!::Th
end
TwiceDifferentiable(f, g!, fg!, h!) = UninitializedTwiceDifferentiable(f, g!, fg!, h!)
TwiceDifferentiable(f, g!, h!) = UninitializedTwiceDifferentiable(f, g!,      nothing, h!)
TwiceDifferentiable(f, g!)     = UninitializedTwiceDifferentiable(f, g!,      nothing, nothing)
TwiceDifferentiable(f)         = UninitializedTwiceDifferentiable(f, nothing, nothing, nothing)
TwiceDifferentiable(u::T, x::AbstractArray) where {T<:UninitializedObjective} = error("Cannot construct a TwiceDifferentiable from UninitializedTwiceDifferentiable unless the gradient and Hessian is provided.")
TwiceDifferentiable(u::UninitializedTwiceDifferentiable, x::AbstractArray) = TwiceDifferentiable(u.f, u.g!, u.fg!, u.h!, x)
TwiceDifferentiable(u::UninitializedTwiceDifferentiable{S, Void, T}, x::AbstractArray) where {S, T} = TwiceDifferentiable(u.f, u.g!, u.h!, x)


NonDifferentiable(f, g!,     x_seed::AbstractArray) = NonDifferentiable(f, x_seed)
NonDifferentiable(f, g!, h!, x_seed::AbstractArray) = NonDifferentiable(f, x_seed)

function OnceDifferentiable(d::OnceDifferentiable, x_seed::AbstractArray)
    value_gradient!(d, x_seed)
    d
end
function OnceDifferentiable(f, x_seed::AbstractArray{T}; autodiff = :finite) where T
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
        error("The autodiff value $autodiff is not supported. Use :finite, :forward or :reverse.")
    end
    g = similar(x_seed)
    g!(g, x_seed)
    return OnceDifferentiable(f, g!, fg!, f(x_seed), g, copy(x_seed), copy(x_seed), f_calls, g_calls)
end

function TwiceDifferentiable(f, x_seed::AbstractArray{T}; autodiff = :finite) where T
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
        gcfg = ForwardDiff.GradientConfig(f, x_seed)
        g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)

        fg! = (out, x) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end

        hcfg = ForwardDiff.HessianConfig(f, x_seed)
        h! = (out, x) -> ForwardDiff.hessian!(out, f, x, hcfg)
    # elseif autodiff == :reverse
    #     gcfg = ReverseDiff.GradientConfig(x_seed)
    #     g! = (out, x) -> ReverseDiff.gradient!(out, f, x, gcfg)
    #
    #     fg! = (out, x) -> begin
    #         gr_res = DiffBase.DiffResult(zero(T), out)
    #         ReverseDiff.gradient!(gr_res, f, x, gcfg)
    #         DiffBase.value(gr_res)
    #     end
    #     hcfg = ReverseDiff.HessianConfig(x_seed)
    #     h! = (out, x) -> ReverseDiff.hessian!(out, f, x, hcfg)
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

function TwiceDifferentiable(f, g!, x_seed::Array{T}; autodiff = :finite) where T
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
TwiceDifferentiable(u::UninitializedTwiceDifferentiable{Void, Void, Void}, x::AbstractArray) = TwiceDifferentiable(u.f, x)
TwiceDifferentiable(u::UninitializedTwiceDifferentiable{T, Void, Void}, x::AbstractArray) where {T} = TwiceDifferentiable(u.f, u.g!, x)
TwiceDifferentiable(d::NonDifferentiable, x_seed::Vector{T} = d.last_x_f; autodiff = :finite) where {T} =
TwiceDifferentiable(d.f, x_seed; autodiff = autodiff)
function TwiceDifferentiable(d::OnceDifferentiable, x_seed::Vector{T} = d.last_x_f; autodiff = :finite) where T
    n_x = length(x_seed)
    if autodiff == :finite
        function h!(storage, x)
            Calculus.finite_difference_hessian!(d.f, x, storage)
            return
        end
    elseif autodiff == :forward
        hcfg = ForwardDiff.HessianConfig(similar(gradient(d)))
        h! = (out, x) -> ForwardDiff.hessian!(out, d.f, x, hcfg)
    # elseif autodiff == :reverse
    #     hcfg = ReverseDiff.HessianConfig(similar(gradient(d)))
    #     h! = (out, x) -> ReverseDiff.hessian!(out, d.f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite, :forward or :reverse.")
    end
    H = Array{T}(n_x, n_x)
    h!(H, d.last_x_g)
    return TwiceDifferentiable(d.f, d.g!, d.fg!, h!, d.f_x,
                               gradient(d), H, d.last_x_f,
                               d.last_x_g, copy(d.last_x_g), d.f_calls, d.g_calls, [1])
end
