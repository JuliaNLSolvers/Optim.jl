

# Standard Preconditioners
#
# Meaning of P:
#    P^{-1} ≈ ∇²E, so the preconditioned gradient is P ∇E
#    this means all functionality can be implemented by
#       P * x        >>> precondfwd!
#       <x|P|y>      >>> precondfwddot
#      <x|P^{-1}|y>  >>> precondinvdot
# and finally an update function >>>>> precondprep!
#     for now, this is passed as an argument to the optimisers (TODO?)

# # default preconditioner update:
# precondprep!(P, x) = nothing


#####################################################
#  [1] Empty preconditioner

# out =  P * A
precondfwd!(out::Array, P::Void, A::Array) = copy!(out, A)
# A' * P * B
precondfwddot(A::Array, P::Void, B::Array) = vecdot(A, B)
# A' * P^{-1} B
precondinvdot(A::Array, P::Void, B::Array) = vecdot(A, B)
# ????? update the preconditioner ?????
precondprep!(P::Void, x) = nothing


#####################################################
#  [2] Diagonal preconditioner

function precondfwd!(out::Array, p::Vector, A::Array)
    @simd for i in 1:length(A)
        @inbounds out[i] = p[i] * A[i]
    end
    return out
end
function precondfwddot{T}(A::Array{T}, p::Vector, B::Array)
    s = zero(T)
    @simd for i in 1:length(A)
        @inbounds s += A[i] * p[i] * B[i]
    end
    return s
end
function precondinvdot{T}(A::Array{T}, p::Vector, B::Array)
    s = zero(T)
    @simd for i in 1:length(A)
        @inbounds s += A[i] * B[i] / p[i]
    end
    return s
end



#####################################################
#  [3] Matrix Preconditioner
#  the assumption here is that P is given by its inverse, which is typical

precondfwd!(out::Vector, P::AbstractMatrix, A::Vector) = copy!(out, P \ A)
precondfwddot(A::Vector, P::AbstractMatrix, B::Vector) = dot(A, P \ B)
precondinvdot(A::Vector, P::AbstractMatrix, B::Vector) = dot(A, P * B)

