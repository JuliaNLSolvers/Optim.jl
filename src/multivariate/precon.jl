

# Some Boiler-plate code for preconditioning
#
# Meaning of P:
#    P ≈ ∇²E, so the preconditioned gradient is P^{-1} ∇E
#    P can be an arbitrary type but at a minimum it MUST provide:
#        A_ldiv_B!(x, P, b) ->  x = P \ b
#        dot(x, P, b)  ->  x' P b
#
#    If `dot` is not provided, then Optim.jl will try to define it via
#        dot(x, P, b) = vecdot(x, A_mul_B!(similar(x), P, y))
#
#    finally the preconditioner can be updated after each x-update using
#        precondprep!
#    but this is passed as an argument at the moment!
#


#####################################################
#  [0] Defaults and aliases for easier reading of the code
#      these can also be over-written if necessary.

# an inner product w.r.t. a metric P (=preconditioner)
Base.dot(x, P, y) = vecdot(x, A_mul_B!(similar(x), P, y))

# default preconditioner update
precondprep!(P, x) = nothing


#####################################################
#  [1] Empty preconditioner = Identity
#

# out =  P^{-1} * A
Base.A_ldiv_B!(out, ::Void, A) = copy!(out, A)
# A' * P B
Base.dot(A, ::Void, B) = vecdot(A, B)


#####################################################
#  [2] Diagonal preconditioner
#      P = Diag(d)
#      unfortunately, Base does not implement
#      A_ldiv_B!(a, P, b) or A_mul_B! for this type, so we do it by hand
#      TODO: maybe implement this in Base
Base.A_ldiv_B!(out::Array, P::Diagonal, A::Array) = copy!(out, A ./ P.diag)
Base.dot(A::Array, P::Diagonal, B::Array) = vecdot(A, P.diag .* B)

#####################################################
#  [3] Inverse Diagonal preconditioner
#      here, P is stored by the entries of its inverse
#      TODO: maybe implement this in Base?
type InverseDiagonal
   diag
end
Base.A_ldiv_B!(out::Array, P::InverseDiagonal, A::Array) = copy!(out, A .* P.diag)
Base.dot(A::Array, P::InverseDiagonal, B::Vector) = vecdot(A, B ./ P.diag)

#####################################################
#  [4] Matrix Preconditioner
#  the assumption here is that P is given by its inverse, which is typical
#     > A_ldiv_B! is about to be moved to Base, so we need a temporary hack
#     > A_mul_B! is already in Base, which defines `dot`
#  nothing to do!
Base.A_ldiv_B!(x, P::AbstractMatrix, b) = copy!(x, P \ b)
