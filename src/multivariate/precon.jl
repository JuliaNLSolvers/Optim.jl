# Some Boiler-plate code for preconditioning
#
# Meaning of P:
#    P ≈ ∇²E, so the preconditioned gradient is P^{-1} ∇E
#    P can be an arbitrary type but at a minimum it MUST provide:
#        ldiv!(x, P, b) ->  x = P \ b
#        dot(x, P, b)  ->  x' P b
#
#    If `dot` is not provided, then Optim.jl will try to define it via
#        dot(x, P, b) = dot(x, mul!(similar(x), P, y))
#
#    finally the preconditioner can be updated after each x-update using
#        precondprep!
#    but this is passed as an argument at the moment!
#

# Fallback
ldiv!(out, M, A) = LinearAlgebra.ldiv!(out, M, A)
dot(a, M, b) = LinearAlgebra.dot(a, M, b)
dot(a, b) = LinearAlgebra.dot(a, b)


#####################################################
#  [0] Defaults and aliases for easier reading of the code
#      these can also be over-written if necessary.

# default preconditioner update
precondprep!(P, x) = nothing


#####################################################
#  [1] Empty preconditioner = Identity
#

# out =  P^{-1} * A
ldiv!(out, ::Nothing, A) = copyto!(out, A)

# A' * P B
dot(A, ::Nothing, B) = dot(A, B)


#####################################################
#  [2] Diagonal preconditioner
#      P = Diag(d)
#      Covered by base

#####################################################
#  [3] Inverse Diagonal preconditioner
#      here, P is stored by the entries of its inverse
#      TODO: maybe implement this in Base?

mutable struct InverseDiagonal
   diag
end
ldiv!(out::AbstractArray, P::InverseDiagonal, A::AbstractArray) = copyto!(out, A .* P.diag)
dot(A::AbstractArray, P::InverseDiagonal, B::Vector) = dot(A, B ./ P.diag)

#####################################################
#  [4] Matrix Preconditioner
#  the assumption here is that P is given by its inverse, which is typical
#     > ldiv! is about to be moved to Base, so we need a temporary hack
#     > mul! is already in Base, which defines `dot`
#  nothing to do!
ldiv!(x, P::AbstractMatrix, b) = copyto!(x, P \ b)
