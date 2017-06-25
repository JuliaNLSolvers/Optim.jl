# Manifold interface: every manifold (subtype of Manifold) defines the functions
# project_tangent!(m, g, x): project g on the tangent space to m at x
# retract!(m, x): map x back to a point on the manifold m

## To add:
## * Second order algorithms
## * Vector transport
## * Arbitrary inner product
## * More retractions
## * More manifolds from ROPTLIB
## * {x, Ax = b}
## * Intersection manifold (just do the projection on both manifolds iteratively and hope it converges)

abstract type Manifold
end


type ManifoldObjective{T<:NLSolversBase.AbstractObjective} <: NLSolversBase.AbstractObjective
    manifold :: Manifold
    inner_obj :: T
end
iscomplex(obj::ManifoldObjective) = iscomplex(obj.inner_obj)
# TODO is it safe here to call retract! and change x?
function NLSolversBase.value!(obj::ManifoldObjective, x)
    xin = complex_to_real(obj, retract(obj.manifold, real_to_complex(obj,x)))
    value!(obj.inner_obj, xin)
end
function NLSolversBase.value(obj::ManifoldObjective)
    value(obj.inner_obj)
end
function NLSolversBase.gradient(obj::ManifoldObjective)
    gradient(obj.inner_obj)
end
function NLSolversBase.gradient(obj::ManifoldObjective,i::Int)
    gradient(obj.inner_obj,i)
end
function NLSolversBase.gradient!(obj::ManifoldObjective,x)
    xin = complex_to_real(obj, retract(obj.manifold, real_to_complex(obj,x)))
    gradient!(obj.inner_obj,xin)
    project_tangent!(obj.manifold,real_to_complex(obj,gradient(obj.inner_obj)),real_to_complex(obj,xin))
    return gradient(obj.inner_obj)
end
function NLSolversBase.value_gradient!(obj::ManifoldObjective,x)
    xin = complex_to_real(obj, retract(obj.manifold, real_to_complex(obj,x)))
    value_gradient!(obj.inner_obj,xin)
    project_tangent!(obj.manifold,real_to_complex(obj,gradient(obj.inner_obj)),real_to_complex(obj,xin))
    return value(obj.inner_obj)
end

# fallback for out-of-place ops
project_tangent(M::Manifold,x) = project_tangent!(M, similar(x), x)
retract(M::Manifold,x) = retract!(M, copy(x))

# Flat manifold = {R,C}^n
# all the functions below are no-ops, and therefore the generated code
# for the flat manifold should be exactly the same as the one with all
# the manifold stuff removed
struct Flat <: Manifold
end
retract(M::Flat, x) = x
retract!(M::Flat,x) = x
project_tangent(M::Flat, g, x) = g
project_tangent!(M::Flat, g, x) = g

# {||x|| = 1}
struct Sphere <: Manifold
end
retract!(S::Sphere, x) = normalize!(x)
project_tangent!(S::Sphere,g,x) = (g .= g .- real(vecdot(x,g)).*x)

# N x n matrices such that X'X = I
# TODO: add more retractions, and support arbitrary inner product
abstract type Stiefel <: Manifold end
struct Stiefel_CholQR <: Stiefel end
struct Stiefel_SVD <: Stiefel end
function Stiefel(retraction=:SVD)
    if retraction == :CholQR
        Stiefel_CholQR()
    elseif retraction == :SVD
        Stiefel_SVD()
    end
end

function retract!(S::Stiefel_SVD, X)
    U,S,V = svd(X)
    X .= U*V'
end
function retract!(S::Stiefel_CholQR, X)
    overlap = X'X
    X .= X/chol(overlap)
end
project_tangent!(S::Stiefel, G, X) = (G .= X*(X'G .- G'X)./2 .+ G .- X*(X'G))



# TODO is there a better way of doing power and product manifolds?

# multiple copies of the same manifold. Points are arrays of arbitrary
# dimensions, and the first (given by inner_dims) are points of the
# inner manifold. E.g. the product of 2x2 Stiefel manifolds of dimension N x n
# would be a N x n x 2 x 2 matrix
struct PowerManifold<:Manifold
    inner_manifold::Manifold #type of embedded manifold
    inner_dims::Tuple #dimension of the embedded manifolds
    outer_dims::Tuple #number of embedded manifolds
end
function retract!(m::PowerManifold, x)
    for i=1:prod(m.outer_dims)
        retract!(m.inner_manifold,get_inner(m, x, i))
    end
    x
end
function project_tangent!(m::PowerManifold, g, x)
    for i=1:prod(m.outer_dims)
        project_tangent!(m.inner_manifold,get_inner(m, g, i),get_inner(m, x, i))
    end
    g
end
# linear indexing
@inline function get_inner(m::PowerManifold, x, i::Int)
    size_inner = prod(m.inner_dims)
    size_outer = prod(m.outer_dims)
    @assert 1 <= i <= size_outer
    return reshape(view(x, (i-1)*size_inner+1:i*size_inner), m.inner_dims)
end
@inline get_inner(m::PowerManifold, x, i::Tuple) = get_inner(m, x, ind2sub(m.outer_dims, i...))

#Product of two manifolds {P = (x1,x2), x1 ∈ m1, x2 ∈ m2}.
#P is assumed to be a flat array, and x1 is before x2 in memory
struct ProductManifold<:Manifold
    m1::Manifold
    m2::Manifold
    dims1::Tuple
    dims2::Tuple
end
function retract!(m::ProductManifold, x)
    retract!(m.m1, get_inner(m,x,1))
    retract!(m.m2, get_inner(m,x,2))
    x
end
function project_tangent!(m::ProductManifold, g, x)
    project_tangent!(m.m1, get_inner(m, g, 1), get_inner(m, x, 1))
    project_tangent!(m.m2, get_inner(m, g, 2), get_inner(m, x, 2))
    g
end
function get_inner(m::ProductManifold, x, i)
    N1 = prod(m.dims1)
    N2 = prod(m.dims2)
    @assert length(x) == N1+N2
    if i == 1
        return reshape(view(x, 1:N1),m.dims1)
    elseif i == 2
        return reshape(view(x, N1+1:N1+N2), m.dims2)
    else
        error("Only two components in a product manifold")
    end
end
