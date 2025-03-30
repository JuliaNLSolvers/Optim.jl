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

# x for updating P
function _precondition!(out, method::AbstractOptimizer, x, ∇f)
    _apply_precondprep(method, x)
    __precondition!(out, method.P, ∇f)
end
# no updating
__precondition!(out, P::Nothing, ∇f) = copyto!(out, ∇f)
# fallback
__precondition!(out, P, ∇f) = ldiv!(out, P, ∇f)
__precondition!(out, P::AbstractMatrix, ∇f) = copyto!(out, P \ ∇f)

function _inverse_precondition(method::AbstractOptimizer, state::AbstractOptimizerState)
    _inverse_precondition(method.P, state.s)
end
function _inverse_precondition(P, s)
    real(dot(s, P, s))
end
function _inverse_precondition(P::Nothing, s)
    real(dot(s, s))
end

_apply_precondprep(method::AbstractOptimizer, x) =
    _apply_precondprep(method.P, method.precondprep!, x)
_apply_precondprep(::Nothing, ::Returns{Nothing}, x) = x
_apply_precondprep(P, precondprep!, x) = precondprep!(P, x)

#####################################################
#  [1] Empty preconditioner = Identity
#
# out =  P^{-1} * A
#####################################################
#  [2] Diagonal preconditioner
#      P = Diag(d)
#      Covered by base
#####################################################
#  [3] Inverse Diagonal preconditioner
#      here, P is stored by the entries of its inverse

mutable struct InverseDiagonal
    diag::Any
end
# If not precondprep was added we just use a constant inverse
_apply_precondprep(P::InverseDiagonal, ::Returns{Nothing}, x) = P
_apply_precondprep(P::InverseDiagonal, precondprep!, x) = precondprep!(P, x)
__precondition!(out, P::InverseDiagonal, ∇f) = copyto!(out, P.diag .* ∇f)

function _inverse_precondition(P::InverseDiagonal, s)
    real(dot(s, P.diag .\ s))
end

#####################################################
#  [4] Matrix Preconditioner
#   Works by stdlib methods
#   It interprets 
