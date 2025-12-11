default_options(method::AbstractOptimizer) = NamedTuple()

fallback_method(f) = NelderMead()
fallback_method(f, g!) = LBFGS()
fallback_method(f, g!, h!) = Newton()

# By default, use central finite difference method
const DEFAULT_AD_TYPE = ADTypes.AutoFiniteDiff(; fdtype = Val(:central))

function fallback_method(f::InplaceObjective)
    if !(f.fdf isa Nothing)
        if !(f.hv isa Nothing)
            return KrylovTrustRegion()
        end
        return LBFGS()
    elseif !(f.fgh isa Nothing)
        return Newton()
    elseif !(f.fghv isa Nothing)
        return KrylovTrustRegion()
    end
end

function fallback_method(f::NotInplaceObjective)
    if !(f.fdf isa Nothing)
        return LBFGS()
    elseif !(f.fgh isa Nothing)
        return LBFGS()
    else
        throw(
            ArgumentError(
                "optimize does not support $(typeof(f)) as the first positional argument",
            ),
        )
    end
end
fallback_method(f::NotInplaceObjective{<:Nothing,<:Nothing,<:Any}) = Newton()

fallback_method(d::OnceDifferentiable) = LBFGS()
fallback_method(d::TwiceDifferentiable) = Newton()

# promote the objective (tuple of callables or an AbstractObjective) according to method requirement
promote_objtype(method, initial_x, autodiff::ADTypes.AbstractADType, inplace::Bool, args...) =
    error("No default objective type for $method and $args.")
# actual promotions, notice that (args...) captures FirstOrderOptimizer and NonDifferentiable, etc
promote_objtype(method::ZerothOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, args...) =
    NonDifferentiable(args..., x, real(zero(eltype(x))))
promote_objtype(method::FirstOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, f) =
    OnceDifferentiable(f, x, real(zero(eltype(x))); autodiff = autodiff)
promote_objtype(method::FirstOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, args...) =
    OnceDifferentiable(args..., x, real(zero(eltype(x))); inplace = inplace)
promote_objtype(method::FirstOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, f, g, h) =
    OnceDifferentiable(f, g, x, real(zero(eltype(x))); inplace = inplace)
promote_objtype(method::SecondOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, f) =
    TwiceDifferentiable(f, x, real(zero(eltype(x))); autodiff = autodiff)
promote_objtype(
    method::SecondOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    f::NotInplaceObjective,
) = TwiceDifferentiable(f, x, real(zero(eltype(x))))
promote_objtype(
    method::SecondOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    f::InplaceObjective,
) = TwiceDifferentiable(f, x, real(zero(eltype(x))))
promote_objtype(method::SecondOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, f, g) =
    TwiceDifferentiable(
        f,
        g,
        x,
        real(zero(eltype(x)));
        inplace = inplace,
        autodiff = autodiff,
    )
promote_objtype(method::SecondOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, f, g, h) =
    TwiceDifferentiable(f, g, h, x, real(zero(eltype(x))); inplace = inplace)
# no-op
promote_objtype(
    method::ZerothOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    nd::NonDifferentiable,
) = nd
promote_objtype(
    method::ZerothOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    od::OnceDifferentiable,
) = od
promote_objtype(
    method::FirstOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    od::OnceDifferentiable,
) = od
promote_objtype(
    method::ZerothOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    td::TwiceDifferentiable,
) = td
promote_objtype(
    method::FirstOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    td::TwiceDifferentiable,
) = td
promote_objtype(
    method::SecondOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    td::TwiceDifferentiable,
) = td

# if no method or options are present
function optimize(
    f,
    initial_x::AbstractArray;
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    method = fallback_method(f)
    d = promote_objtype(method, initial_x, autodiff, inplace, f)

    options = Options(; default_options(method)...)
    optimize(d, initial_x, method, options)
end
function optimize(
    f,
    g,
    initial_x::AbstractArray;
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
    inplace::Bool = true,
)

    method = fallback_method(f, g)

    d = promote_objtype(method, initial_x, autodiff, inplace, f, g)
 
    options = Options(; default_options(method)...)
    optimize(d, initial_x, method, options)
end
function optimize(
    f,
    g,
    h,
    initial_x::AbstractArray;
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE, 
)
    method = fallback_method(f, g, h)
    d = promote_objtype(method, initial_x, autodiff, inplace, f, g, h)

    options = Options(; default_options(method)...)
    optimize(d, initial_x, method, options)
end

# no method supplied with objective
function optimize(
    d::T,
    initial_x::AbstractArray,
    options::Options,
) where {T<:AbstractObjective}
    optimize(d, initial_x, fallback_method(d), options)
end
# no method supplied with inplace and autodiff keywords becauase objective is not supplied
function optimize(
    f,
    initial_x::AbstractArray,
    options::Options;
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    method = fallback_method(f)
    d = promote_objtype(method, initial_x, autodiff, inplace, f)
    optimize(d, initial_x, method, options)
end
function optimize(
    f,
    g,
    initial_x::AbstractArray,
    options::Options;
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)

    method = fallback_method(f, g)
    d = promote_objtype(method, initial_x, autodiff, inplace, f, g)
    optimize(d, initial_x, method, options)
end
function optimize(
    f,
    g,
    h,
    initial_x::AbstractArray{T},
    options::Options;
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
) where {T}
    method = fallback_method(f, g, h)
    d = promote_objtype(method, initial_x, autodiff, inplace, f, g, h)

    optimize(d, initial_x, method, options)
end

# potentially everything is supplied (besides caches)
function optimize(
    f,
    initial_x::AbstractArray,
    method::AbstractOptimizer,
    options::Options = Options(; default_options(method)...);
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    d = promote_objtype(method, initial_x, autodiff, inplace, f)
    optimize(d, initial_x, method, options)
end
function optimize(
    f,
    c::AbstractConstraints,
    initial_x::AbstractArray,
    method::AbstractOptimizer,
    options::Options = Options(; default_options(method)...);
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)

    d = promote_objtype(method, initial_x, autodiff, inplace, f)
    optimize(d, c, initial_x, method, options)
end
function optimize(
    f,
    g,
    initial_x::AbstractArray,
    method::AbstractOptimizer,
    options::Options = Options(; default_options(method)...);
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    d = promote_objtype(method, initial_x, autodiff, inplace, f, g)

    optimize(d, initial_x, method, options)
end
function optimize(
    f,
    g,
    h,
    initial_x::AbstractArray,
    method::AbstractOptimizer,
    options::Options = Options(; default_options(method)...);
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
   
)
    d = promote_objtype(method, initial_x, autodiff, inplace, f, g, h)

    optimize(d, initial_x, method, options)
end

function optimize(
    d::D,
    initial_x::AbstractArray,
    method::SecondOrderOptimizer,
    options::Options = Options(; default_options(method)...);
    inplace::Bool = true,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
) where {D<:Union{NonDifferentiable,OnceDifferentiable}}
    d = promote_objtype(method, initial_x, autodiff, inplace, d)
    optimize(d, initial_x, method, options)
end
