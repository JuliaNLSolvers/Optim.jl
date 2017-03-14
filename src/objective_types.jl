@compat abstract type AbstractObjective end
type NonDifferentiable{T} <: AbstractObjective
    f
    f_x::T
    last_x_f::Array{T}
    f_calls::Vector{Int}
end
NonDifferentiable{T}(f, x_seed::Array{T}) = NonDifferentiable(f, f(x_seed), copy(x_seed), [1])

type OnceDifferentiable{T, Tgrad} <: AbstractObjective
    f
    g!
    fg!
    f_x::T
    g::Tgrad
    last_x_f::Array{T}
    last_x_g::Array{T}
    f_calls::Vector{Int}
    g_calls::Vector{Int}
end
function OnceDifferentiable(f, g!, fg!, x_seed)
    g = similar(x_seed)
    g!(x_seed, g)
    OnceDifferentiable(f, g!, fg!, f(x_seed), g, copy(x_seed), copy(x_seed), [1], [1])
end
function OnceDifferentiable(f, g!, x_seed)
    function fg!(x, storage)
        g!(x, storage)
        return f(x)
    end
    return OnceDifferentiable(f, g!, fg!, x_seed)
end
function OnceDifferentiable{T}(f, x_seed::Vector{T}; autodiff = :finite)
    n_x = length(x_seed)
    f_calls = [1]
    g_calls = [1]
    if autodiff == :finite
        function g!(x, storage)
            Calculus.finite_difference!(f, x, storage, :central)
            return
        end
        function fg!(x, storage)
            g!(x, storage)
            return f(x)
        end
    elseif autodiff == :forward
        gcfg = ForwardDiff.GradientConfig(x_seed)
        g! = (x, out) -> ForwardDiff.gradient!(out, f, x, gcfg)

        fg! = (x, out) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end
    elseif autodiff == :reverse
        gcfg = ReverseDiff.GradientConfig(x_seed)
        g! = (x, out) -> ReverseDiff.gradient!(out, f, x, gcfg)

        fg! = (x, out) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ReverseDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end
    else
        error("The autodiff value $autodiff is not supported. Use :finite, :forward or :reverse.")
    end
    g = similar(x_seed)
    g!(x_seed, g)
    return OnceDifferentiable(f, g!, fg!, f(x_seed), g, copy(x_seed), copy(x_seed), f_calls, g_calls)
end

type TwiceDifferentiable{T<:Real} <: AbstractObjective
    f
    g!
    fg!
    h!
    f_x::T
    g::Vector{T}
    H::Matrix{T}
    last_x_f::Vector{T}
    last_x_g::Vector{T}
    last_x_h::Vector{T}
    f_calls::Vector{Int}
    g_calls::Vector{Int}
    h_calls::Vector{Int}
end
function TwiceDifferentiable{T}(f, g!, fg!, h!, x_seed::Array{T})
    n_x = length(x_seed)
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)
    g!(x_seed, g)
    h!(x_seed, H)
    TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                g, H, copy(x_seed),
                                copy(x_seed), copy(x_seed), [1], [1], [1])
end

function TwiceDifferentiable{T}(f,
                                 g!,
                                 h!,
                                 x_seed::Array{T})
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    return TwiceDifferentiable(f, g!, fg!, h!, x_seed)
end
function TwiceDifferentiable{T}(f, x_seed::Vector{T}; autodiff = :finite)
    n_x = length(x_seed)
    f_calls = [1]
    g_calls = [1]
    h_calls = [1]
    if autodiff == :finite
        function g!(x::Vector, storage::Vector)
            Calculus.finite_difference!(f, x, storage, :central)
            return
        end
        function fg!(x::Vector, storage::Vector)
            g!(x, storage)
            return f(x)
        end
        function h!(x::Vector, storage::Matrix)
            Calculus.finite_difference_hessian!(f, x, storage)
            return
        end
    elseif autodiff == :forward
        gcfg = ForwardDiff.GradientConfig(x_seed)
        g! = (x, out) -> ForwardDiff.gradient!(out, f, x, gcfg)

        fg! = (x, out) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end

        hcfg = ForwardDiff.HessianConfig(x_seed)
        h! = (x, out) -> ForwardDiff.hessian!(out, f, x, hcfg)
    elseif autodiff == :reverse
        gcfg = ReverseDiff.GradientConfig(x_seed)
        g! = (x, out) -> ReverseDiff.gradient!(out, f, x, gcfg)

        fg! = (x, out) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ReverseDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end
        hcfg = ReverseDiff.HessianConfig(x_seed)
        h! = (x, out) -> ReverseDiff.hessian!(out, f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite, :forward or :reverse.")
    end
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)
    g!(x_seed, g)
    h!(x_seed, H)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g, H, copy(x_seed),
                                       copy(x_seed), copy(x_seed), f_calls, g_calls, h_calls)
end


function TwiceDifferentiable{T}(f, g!, x_seed::Array{T}; autodiff = :finite)
    n_x = length(x_seed)
    f_calls = [1]
    function fg!(x, storage)
        g!(x, storage)
        return f(x)
    end
    if autodiff == :finite
        function h!(x, storage)
            Calculus.finite_difference_hessian!(f, x, storage)
            return
        end
    elseif autodiff == :forward
        hcfg = ForwardDiff.HessianConfig(similar(x_seed))
        h! = (x, out) -> ForwardDiff.hessian!(out, f, x, hcfg)
    elseif autodiff == :reverse
        hcfg = ReverseDiff.HessianConfig(x_seed)
        h! = (x, out) -> ReverseDiff.hessian!(out, f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite, :forward or :reverse.")
    end
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)
    g!(x_seed, g)
    h!(x_seed, H)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g, H, copy(x_seed),
                                       copy(x_seed), copy(x_seed), f_calls, [1], [1])
end
#=
function TwiceDifferentiable{T}(f, g!, fg!, x_seed::Array{T}; autodiff = :finite)
    n_x = length(x_seed)
    f_calls = [1]
    if autodiff == :finite
        function h!(x::Vector, storage::Matrix)
            Calculus.finite_difference_hessian!(f, x, storage)
            return
        end
    elseif autodiff == :forward
        hcfg = ForwardDiff.HessianConfig(similar(gradient(d)))
        h! = (x, out) -> ForwardDiff.hessian!(out, f, x, hcfg)
    end
    g = similar(x_seed)
    g!(x_seed, g)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g, Array{T}(n_x, n_x), copy(x_seed), f_calls, [1], [0])
end
=#
function TwiceDifferentiable(d::OnceDifferentiable; autodiff = :finite)
    n_x = length(d.last_x_f)
    T = eltype(d.last_x_f)
    if autodiff == :finite
        function h!(x::Vector, storage::Matrix)
            Calculus.finite_difference_hessian!(d.f, x, storage)
            return
        end
    elseif autodiff == :forward
        hcfg = ForwardDiff.HessianConfig(similar(gradient(d)))
        h! = (x, out) -> ForwardDiff.hessian!(out, d.f, x, hcfg)
    end
    H = Array{T}(n_x, n_x)
    h!(d.last_x_g, H)
    return TwiceDifferentiable(d.f, d.g!, d.fg!, h!, d.f_x,
                                       gradient(d), H, d.last_x_f,
                                       d.last_x_g, copy(d.last_x_g), d.f_calls, d.g_calls, [1])
end

function _unchecked_value!(obj, x)
    obj.f_calls .+= 1
    copy!(obj.last_x_f, x)
    obj.f_x = obj.f(x)
end
function value(obj, x)
    if x != obj.last_x_f
        obj.f_calls += 1
        return obj.f(x)
    end
    obj.f_x
end
function value!(obj, x)
    if x != obj.last_x_f
        _unchecked_value!(obj, x)
    end
    obj.f_x
end


function _unchecked_grad!(obj, x)
    obj.g_calls .+= 1
    copy!(obj.last_x_g, x)
    obj.g!(x)
end
function gradient!(obj, x)
    if x != obj.last_x_g
        _unchecked_gradient!(obj, x)
    end
end

function value_grad!(obj, x)
    if x != obj.last_x_f && x != obj.last_x_g
        obj.f_calls .+= 1
        obj.g_calls .+= 1
        obj.last_x_f[:], obj.last_x_g[:] = copy(x), copy(x)
        obj.f_x = obj.fg!(x, obj.g)
    elseif x != obj.last_x_f
        _unchecked_value!(obj, x)
    elseif x != obj.last_x_g
        _unchecked_grad!(obj, x)
    end
    obj.f_x
end

function _unchecked_hessian!(obj, x)
    obj.h_calls .+= 1
    copy!(obj.last_x_h, x)
    obj.h!(x, obj.H)
end
function hessian!(obj, x)
    if x != obj.last_x_h
        _unchecked_hessian!(obj, x)
    end
end

# Getters are without ! and accept only an objective and index or just an objective
value(obj) = obj.f_x
gradient(obj) = obj.g
gradient(obj, i::Integer) = obj.g[i]
hessian(obj) = obj.H
