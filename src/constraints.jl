abstract AbstractConstraints

immutable ConstraintsNone <: AbstractConstraints end

immutable ConstraintsBox{T,N} <: AbstractConstraints
    lower::Array{T,N}
    upper::Array{T,N}

    function ConstraintsBox(l::AbstractArray{T}, u::AbstractArray{T})
        size(l) == size(u) || error("The sizes of the bounds must match")
        for i = 1:length(l)
            l[i] <= u[i] || error("The lower bound must be smaller than the upper bound")
        end
        new(l, u)
    end
end
ConstraintsBox{T,N}(l::AbstractArray{T,N}, u::AbstractArray{T,N}) = ConstraintsBox{T,N}(l, u)
ConstraintsBox{T,N}(l::AbstractArray{T,N}, u::Nothing) = ConstraintsBox{T,N}(l, infs(T, size(l)))
ConstraintsBox{T,N}(l::Nothing, u::AbstractArray{T,N}) = ConstraintsBox{T,N}(-infs(T,size(u)), u)

immutable ConstraintsL{T,M<:AbstractMatrix,N} <: AbstractConstraints
    A::M
    lower::Array{T,N}
    upper::Array{T,N}
    scratch1::Array{T,N}
    scratch2::Array{T,N}
    scratch3::Array{T,N}

    function ConstraintsL(A, l::Array{T,N}, u::Array{T,N},
                          scratch1 = similar(l), scratch2 = similar(l), scratch3 = Array(T, ndims(l) == 1 ? size(A, 2) : (size(A,2),size(l,2))))
        size(A, 1) == size(l,1) == size(u,1) || error("The sizes of the bounds must match the size of A")
        for i = 1:length(l)
            l[i] <= u[i] || error("The lower bound must be smaller than the upper bound")
        end
        new(A, l, u, scratch1, scratch2, scratch3)
    end
end
ConstraintsL{T}(A::AbstractMatrix, l::Union(Vector{T},Matrix{T}), u::Union(Vector{T},Matrix{T})) = ConstraintsL{T,typeof(A),ndims(l)}(A, l, u)
ConstraintsL{T}(A::AbstractMatrix, l::Union(Vector{T},Matrix{T}), u::Nothing) = ConstraintsL(A, l, infs(T, size(l)))
ConstraintsL{T}(A::AbstractMatrix, l::Nothing, u::Union(Vector{T},Matrix{T})) = ConstraintsL(A, -infs(T,size(u)), u)

# Generic constraints
immutable ConstraintsNL{T,F<:Union(Function,DifferentiableFunction,TwiceDifferentiableFunction)} <: AbstractConstraints
    funcs::Vector{F}
    lower::Vector{T}
    upper::Vector{T}

    function ConstraintsNL(funcs::Vector, l::AbstractVector{T}, u::AbstractVector{T})
        size(A, 1) == length(l) == length(u) || error("The sizes of the bounds must match the ")
        for i = 1:length(l)
            l[i] <= u[i] || error("The lower bound must be smaller than the upper bound")
        end
        new(funcs, l, u)
    end
end

type Constraints
    bounds::AbstractConstraints
    linear::AbstractConstraints
    nonlinear::AbstractConstraints
    mu::Real
end
Constraints() = Constraints(ConstraintsNone(), ConstraintsNone(), ConstraintsNone(), ConstraintsNone(), ConstraintsNone())


## feasible
feasible(x, constraints::ConstraintsNone, ineq::Bool) = true

function feasible(x, constraints::ConstraintsBox)
    l = constraints.lower
    u = constraints.upper
    for i = 1:length(x)
        l[i] <= x[i] <= u[i] || return false
    end
    true
end

function feasible(x, constraints::ConstraintsL)
    l = constraints.lower
    u = constraints.upper
    y = constraints.scratch1
    A_mul_B!(y, constraints.A, x)
    for i = 1:length(y)
        l[i] <= y[i] <= u[i] || return false
    end
    true
end

## project!
project!(x, constraints::AbstractConstraints) = x

function project!(x::Array, bounds::ConstraintsBox)
    for i = 1:length(x)
        x[i] = max(bounds.lower[i], min(bounds.upper[i], x[i]))
    end
    x
end

## step!
function step!{T}(xtmp::Array{T}, x, s, alpha::Real, constraints::AbstractConstraints = ConstraintsNone{T}())
    (length(xtmp) == length(x) == length(s)) || error("lengths of xtmp ($(length(xtmp))), x ($(length(x))), and s ($(length(s))) disagree")
    for i = 1:length(xtmp)
        @inbounds xtmp[i] = x[i] + alpha*s[i]
    end
    project!(xtmp, constraints)
end


## toedge and tocorner
# For bounds constraints,
#  - toedge returns the "distance along s" (the value of the coefficient alpha) to reach the boundary
#  - tocorner returns the "distance along s" needed to be projected into the "corner" (all constraints are active)
# The first is the farthest you might consider searching in a barrier method. The second is the farthest that it
# makes sense to search in a projection method.
toedge{T}(x::AbstractArray{T}, s, constraints::AbstractConstraints) = inf(T)
tocorner{T}(x::AbstractArray{T}, s, constraints::AbstractConstraints) = inf(T)

function toedge{T}(x::AbstractArray{T}, s, bounds::ConstraintsBox)
    # Stop at the first coordinate to leave the box
    alphamax = inf(T)
    for i = 1:length(x)
        si = s[i]
        li = bounds.lower[i]
        ui = bounds.upper[i]
        if si < 0 && isfinite(li)
            alphamax = min(alphamax, (li-x[i])/si)
        elseif si > 0 && isfinite(ui)
            alphamax = min(alphamax, (ui-x[i])/si)
        end
    end
    convert(T, alphamax)
end

toedge(x, s, constraints::Constraints) = toedge(x, s, constraints.bounds)

function tocorner{T}(x::AbstractArray{T}, s, bounds::ConstraintsBox)
    # Stop at the last coordinate to leave the box
    alphamax = zero(T)
    for i = 1:length(x)
        si = s[i]
        li = bounds.lower[i]
        ui = bounds.upper[i]
        if si < 0 && isfinite(li)
            alphamax = max(alphamax, (li-x[i])/si)
        elseif si > 0 && isfinite(ui)
            alphamax = max(alphamax, (ui-x[i])/si)
        else
            return inf(T)
        end
    end
    convert(T, alphamax)
end

tocorner(x, s, constraints::Constraints) = tocorner(x, s, constraints.bounds)
