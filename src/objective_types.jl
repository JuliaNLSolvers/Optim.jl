function OnceDifferentiable(f, x_seed::AbstractArray{T}, F::Real = real(zero(T)),
                            DF::AbstractArray = NLSolversBase.alloc_DF(x_seed, F), ::Val{S} = Val{true}();
                            autodiff = :finite) where {T,S}
    if autodiff == :finite
        # TODO: Allow user to specify Val{:central}, Val{:forward}, :Val{:complex} (requires care when using :forward I think)
        gcache = DiffEqDiffTools.GradientCache(x_seed, x_seed, Val{:central})
        function g!(storage, x)
            DiffEqDiffTools.finite_difference_gradient!(storage, f, x, gcache)
            return
        end
        function fg!(storage, x)
            g!(storage, x)
            return f(x)
        end
    elseif autodiff == :forward
        gcfg = ForwardDiff.GradientConfig(f, x_seed, ForwardDiff.Chunk(x_seed))
        g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)
        gr_res = DiffResults.MutableDiffResult(zero(T), (x_seed,))
        fg! = (out, x) -> begin
            gr_res.derivs = (out,)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffResults.value(gr_res)
        end
    else
        error("The autodiff value $autodiff is not support. Use :finite or :forward.")
    end
    OnceDifferentiable(f, g!, fg!, x_seed, F, DF, Val{S}())
end
# Val{:central} default?
function OnceDifferentiable(f, x_seed::AbstractArray{T}, F, ::Val{finite}, ::Val{S} ) where {T, finite, S}
    gcache = DiffEqDiffTools.GradientCache(x_seed, x_seed, Val{finite}) #Replace with Val instance when 0.6 support dropped
    function g!(storage, x)
        DiffEqDiffTools.finite_difference_gradient!(storage, f, x, gcache)
        return
    end
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    OnceDifferentiable(f, g!, fg!, x_seed, F, Val{S}())
end

# gr_res seems stable through the closures on 0.7. @codewarntype on od.fdf shows:
#  #self#::getfield(Main, Symbol("##112#114")){typeof(rosenbrock),ForwardDiff.GradientConfig{ForwardDiff.Tag{typeof(rosenbrock),Float64},Float64,2,Array{ForwardDiff.Dual{ForwardDiff.Tag{typeof(rosenbrock),Float64},Float64,2},1}},DiffResults.MutableDiffResult{1,Float64,Tuple{Array{Float64,1}}}}
#   out::Array{Float64,1}
#   x::Array{Float64,1}
# Creation of OnceDifferentiable object for Val(:forward) is not type stable, because of ForwardDiff.Chunk(x_seed).
# I am now just exposing that API here, to make it easier to specify manually through Optim.
# Odds are though, that if someone really wants to configure their differentiation, they'd just construct
# the once differentiable object with an f, g!, and fg! themselves.
# I think pointing people that way would make more sense than adding this to Optim.Options?
# 
# Instability in gcfg contaminates g! and fg!
# Therefore, it may be a good idea to have Val(:forward) default to creating untyped OnceDifferentiables.
function OnceDifferentiable(f, x_seed::AbstractArray{T}, F, ::Val{:forward}, ::Val{S}, ::ForwardDiff.Chunk{N} = ForwardDiff.Chunk(x_seed)) where {T, S, N}
    gcfg = ForwardDiff.GradientConfig(f, x_seed, ForwardDiff.Chunk{N}())
    g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)
    gr_res = DiffResults.MutableDiffResult(zero(T), (x_seed,))
    fg! = (out, x) -> begin
        gr_res.derivs = (out,)
        ForwardDiff.gradient!(gr_res, f, x, gcfg)
        DiffResults.value(gr_res)
    end
    OnceDifferentiable(f, g!, fg!, x_seed, F, Val{S}())
end

function TwiceDifferentiable(f, g!, x_seed::AbstractVector{T}, F::Real = real(zero(T)); autodiff = :finite) where T
    n_x = length(x_seed)
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    if autodiff == :finite
        # TODO: Create / request Hessian functionality in DiffEqDiffTools?
        #       (Or is it better to use the finite difference Jacobian of a gradient?)
        # TODO: Allow user to specify Val{:central}, Val{:forward}, :Val{:complex}
        jcache = DiffEqDiffTools.JacobianCache(x_seed, Val{:central})
        function h!(storage, x)
            DiffEqDiffTools.finite_difference_jacobian!(storage, g!, x, jcache)
            return
        end
    elseif autodiff == :forward
        hcfg = ForwardDiff.HessianConfig(f, similar(x_seed))
        h! = (out, x) -> ForwardDiff.hessian!(out, f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite or :forward.")
    end
    TwiceDifferentiable(f, g!, fg!, h!, x_seed, F)
end

function TwiceDifferentiable(f, g!, x_seed::AbstractVector{T}, F::Real, ::Val{finite}, ::Val{S}) where {T,finite,S}
    n_x = length(x_seed)
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end

    # TODO: Create / request Hessian functionality in DiffEqDiffTools?
    #       (Or is it better to use the finite difference Jacobian of a gradient?)
    # TODO: Allow user to specify Val{:central}, Val{:forward}, :Val{:complex}
    jcache = DiffEqDiffTools.JacobianCache(x_seed, Val{finite})
    function h!(storage, x)
        DiffEqDiffTools.finite_difference_jacobian!(storage, g!, x, jcache)
        return
    end
    TwiceDifferentiable(f, g!, fg!, h!, x_seed, F, Val{S}())
end

function TwiceDifferentiable(f, g!, x_seed::AbstractVector{T}, F::Real, ::Val{:forward}, ::Val{S}) where {T,S}
    n_x = length(x_seed)
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    hcfg = ForwardDiff.HessianConfig(similar(x_seed))
    h! = (out, x) -> ForwardDiff.hessian!(out, f, x, hcfg)
    TwiceDifferentiable(f, g!, fg!, h!, x_seed, F, Val{S}())
end

TwiceDifferentiable(d::NonDifferentiable, x_seed::AbstractVector{T} = d.x_f, F::Real = real(zero(T)); autodiff = :finite) where {T<:Real} =
    TwiceDifferentiable(d.f, x_seed, F; autodiff = autodiff)

function TwiceDifferentiable(d::OnceDifferentiable, x_seed::AbstractVector{T} = d.x_f,
                             F::Real = real(zero(T)); autodiff = :finite) where T<:Real
    if autodiff == :finite
        # TODO: Create / request Hessian functionality in DiffEqDiffTools?
        #       (Or is it better to use the finite difference Jacobian of a gradient?)
        # TODO: Allow user to specify Val{:central}, Val{:forward}, :Val{:complex}
        jcache = DiffEqDiffTools.JacobianCache(x_seed, Val{:central})
        function h!(storage, x)
            DiffEqDiffTools.finite_difference_jacobian!(storage, d.df, x, jcache)
            return
        end
    elseif autodiff == :forward
        hcfg = ForwardDiff.HessianConfig(similar(gradient(d)))
        h! = (out, x) -> ForwardDiff.hessian!(out, d.f, x, hcfg)
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite or :forward.")
    end
    return TwiceDifferentiable(d.f, d.df, d.fdf, h!, x_seed, F, gradient(d))
end

function TwiceDifferentiable(f, x::AbstractVector{T}, F::Real = real(zero(T)), ::Val{S} = Val{true}();
                             autodiff = :finite) where {S, T}
    if autodiff == :finite
        # TODO: Allow user to specify Val{:central}, Val{:forward}, Val{:complex}
        gcache = DiffEqDiffTools.GradientCache(x, x, Val{:central})
        function g!(storage, x)
            DiffEqDiffTools.finite_difference_gradient!(storage, f, x, gcache)
            return
        end
        function fg!(storage::Vector, x::Vector)
            g!(storage, x)
            return f(x)
        end
        # TODO: Allow user to specify Val{:central}, Val{:forward}, :Val{:complex}
        function h!(storage::Matrix, x::Vector)
            # TODO: Wait to use DiffEqDiffTools until they introduce the Hessian feature
            Calculus.finite_difference_hessian!(f, x, storage)
            return
        end
    elseif autodiff == :forward
        gcfg = ForwardDiff.GradientConfig(f, x)
        hcfg = ForwardDiff.HessianConfig(f, x)
        g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)
        h_res = DiffResults.MutableDiffResult(zero(T), (x,Matrix{T}(undef,0,0)))
        fg! = (out, x) -> begin
            h_res.derivs = ( out, h_res.derivs[2] )
            ForwardDiff.gradient!(h_res, f, x, hcfg)
            DiffResults.value(h_res)
        end
    
        h! = (out, x) -> begin
            h_res.derivs = ( h_res.derivs[1], out )
            ForwardDiff.hessian!(h_res, f, x, hcfg)
            out
        end
        TwiceDifferentiable(f, g!, fg!, h!, x, F, Val{S}())
    else
        error("The autodiff value $(autodiff) is not supported. Use :finite or :forward.")
    end
    TwiceDifferentiable(f, g!, fg!, h!, x, F)
end

function TwiceDifferentiable(f, x::AbstractVector{T}, F::Real, ::Val{finite}, ::Val{S}) where {T,finite,S}
    # TODO: Allow user to specify Val{:central}, Val{:forward}, Val{:complex}
    # currently this is only accessible when a user is calling TwiceDifferentiable directly.
    gcache = DiffEqDiffTools.GradientCache(x, x, Val{finite})
    function g!(storage, x)
        DiffEqDiffTools.finite_difference_gradient!(storage, f, x, gcache)
        return
    end
    function fg!(storage::Vector, x::Vector)
        g!(storage, x)
        return f(x)
    end
    # TODO: Allow user to specify Val{:central}, Val{:forward}, :Val{:complex}
    function h!(storage::Matrix, x::Vector)
        # TODO: Wait to use DiffEqDiffTools until they introduce the Hessian feature
        Calculus.finite_difference_hessian!(f, x, storage)
        return
    end
    TwiceDifferentiable(f, g!, fg!, h!, x, F, Val{S}())
end

function TwiceDifferentiable(f, x::AbstractVector{T}, F::Real, ::Val{:forward}, ::Val{S}, ::ForwardDiff.Chunk{N} = ForwardDiff.Chunk(x)) where {T,S,N}
    gcfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{N}())
    hcfg = ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{N}())
    g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)
    g_res = DiffResults.MutableDiffResult(zero(T), (x, ))
    h_res = DiffResults.HessianResult(x)
    fg! = (out, x) -> begin
        g_res.derivs = ( out, )
        ForwardDiff.gradient!(g_res, f, x, gcfg)
        DiffResults.value(g_res)
    end

    h! = (out, x) -> ForwardDiff.hessian!(out, f, x, hcfg)
    TwiceDifferentiable(f, g!, fg!, h!, x, F, Val{S}())
end
