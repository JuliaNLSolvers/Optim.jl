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
#  [1] Empty preconditioner = Identity
#
# out =  P^{-1} * A
ldiv!(out, ::Nothing, A) = copyto!(out, A)


#####################################################
#  [2] Diagonal preconditioner
#      P = Diag(d)
#      Covered by base

#####################################################
#  [3] Inverse Diagonal preconditioner
#      here, P is stored by the entries of its inverse
#      TODO: maybe implement this in Base?

mutable struct InverseDiagonal
    diag::Any
end
ldiv!(out::AbstractArray, P::InverseDiagonal, A::AbstractArray) = copyto!(out, A .* P.diag)
dot(A::AbstractArray, P::InverseDiagonal, B::Vector) = dot(A, B ./ P.diag)

#####################################################
#  [4] Matrix Preconditioner
# Works by stdlib methods

_apply_precondprep(method::AbstractOptimizer, x) =
    _apply_precondprep(method.P, method.precondprep!, x)
_apply_precondprep(::Nothing, precondprep!, x) = x
_apply_precondprep(P, precondprep!, x) = precondprep!(P, x)

_precond_dot(method::AbstractOptimizer, state::AbstractOptimizerState) =
    _precond_dot(state.s, method.P, state.s)
_precond_dot(s, P, p) = real(dot(s, P, s))
_precond_dot(s, P::Nothing, p) = real(dot(s, s))
