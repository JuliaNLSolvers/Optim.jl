# Manifold interface: every manifold (subtype of Manifold) defines the functions
# project_tangent(m, g, x): project g on the tangent space to m at x
# retract(m, x): map x back to a point on the manifold m

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
function NLSolversBase.value_gradient!(obj::ManifoldObjective,x)
    xin = retract(obj.manifold, x)
    value_gradient!(obj.inner_obj,xin)
    project_tangent!(obj.manifold,gradient(obj.inner_obj),xin)
    return value(obj.inner_obj)
end

# fallback for in-place ops
project_tangent!(M::Manifold,g,x) = copy!(g,project_tangent(M,g,x))
retract!(M::Manifold,x) = copy!(x,retract(M,x))


# Flat manifold = {R,C}^n
# all the functions below are no-ops, and therefore the generated code
# for the flat manifold should be exactly the same as the one with all
# the manifold stuff removed
struct Flat <: Manifold
end
retract(M::Flat, x) = x
project_tangent(M::Flat, g, x) = g
project_tangent!(M::Flat, g, x) = g
retract!(M::Flat,x) = x
ManifoldObjective(m::Flat, obj) = obj

# {||x|| = 1}
struct Sphere <: Manifold
end
retract(S::Sphere, x) = x/vecnorm(x)
project_tangent(S::Sphere,g,x) = g - real(vecdot(x,g))*x

# Nxn matrices such that X'X = I
struct Stiefel <: Manifold
end
function retract(S::Stiefel, X)
    U,S,V = svd(X)
    return U*V'
end
project_tangent(S::Stiefel, G, X) = X*(X'G .- G'X)./2 .+ G .- X*(X'G)
