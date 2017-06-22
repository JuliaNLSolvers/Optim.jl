# Manifold interface: every manifold (subtype of Manifold) defines the functions
# project_tangent!(m, g, x): project g on the tangent space to m at x
# retract!(m, x): map x back to a point on the manifold m

## To add:
## * Second order algorithms
## * Vector transport
## * More retractions
## * More manifolds from ROPTLIB
## * {x, Ax = b}
## * Intersection manifold (just do the projection on both manifolds iteratively and hope it converges)

abstract type Manifold
end


type ManifoldObjective{T<:AbstractObjective}
    manifold :: Manifold
    inner_obj :: T
end
function NLSolversBase.value!(obj::ManifoldObjective, x)
    value!(obj.inner_obj, retract(obj.manifold, x))
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
function NLSolversBase.value_gradient!(obj::ManifoldObjective,x)
    xin = retract(obj.manifold, x)
    value_gradient!(obj.inner_obj,xin)
    project_tangent!(obj.manifold,gradient(obj.inner_obj),xin)
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
ManifoldObjective(m::Flat, obj) = obj #not strictly necessary

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
# inner manifold. E.g. the product of two N x n Stiefel manifolds
# would be a N x n x 2 matrix
struct PowerManifold<:Manifold
    inner_manifold::Manifold #type of manifold
    inner_dims::Tuple #dimension of the embedded manifolds
    ninner::Int #number of manifolds
end
function retract!(m::PowerManifold, x)
    size_inner = prod(m.inner_dims)
    @assert length(x) == size_inner*m.ninner
    for i=1:m.ninner
        @views retract!(m.inner_manifold, reshape(x[(i-1)*size_inner+1:i*size_inner], m.inner_dims))
    end
    x
end
function project_tangent!(m::PowerManifold, g, x)
    size_inner = prod(m.inner_dims)
    @assert length(g) == length(x) == size_inner*m.ninner
    for i=1:m.ninner
        @views project_tangent!(m.inner_manifold,
                                reshape(g[(i-1)*size_inner+1:i*size_inner], m.inner_dims),
                                reshape(x[(i-1)*size_inner+1:i*size_inner], m.inner_dims))
    end
    g
end

#Product of two manifolds {P = (x1,x2), x1 ∈ m1, x2 ∈ m2}.
#P is assumed to be a flat array, and x1 is before x2 in memory
struct ProductManifold<:Manifold
    m1::Manifold
    m2::Manifold
    dims1::Tuple
    dims2::Tuple
end
@views function retract!(m::ProductManifold, x)
    N1 = prod(m.dims1)
    N2 = prod(m.dims2)
    @assert N1+N2 == length(x)
    retract!(m.m1, x[1:N1])
    retract!(m.m2, x[N1+1:N1+N2])
    x
end
@views function project_tangent!(m::ProductManifold, g, x)
    N1 = prod(m.dims1)
    N2 = prod(m.dims2)
    @assert N1+N2 == length(x) == length(g)
    project_tangent!(m.m1, g[1:N1], x[1:N1])
    project_tangent!(m.m2, g[N1+1:N1+N2], x[N1+1:N1+N2])
    g
end
