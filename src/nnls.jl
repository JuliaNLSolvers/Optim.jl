function nnls(A::AbstractMatrix, b::AbstractVector, ops::Options)
    # Set up the preconditioner as the inverse diagonal of the Hessian
    a = sum(A.^2, 1)
    @defaults ops precondprep=(P, x, l, u, mu)->precondprepnnls(P, x, mu, a)
    ops = copy(ops)
    @set_options ops precondprep=precondprep
    # Create the initial guess (an interior point)
    T = promote_type(eltype(A), eltype(b))
    x = fill(one(T), size(A, 2))
    # Set up constraints
    l = zeros(eltype(x), length(x))
    u = fill(inf(eltype(x)), length(x))
    # Perform the optimization    
    func = (g,x) -> nnlsobjective(g, x, A, b)
    fminbox(func, x, l, u, ops)
end
nnls(A::AbstractMatrix, b::AbstractVector) = nnls(A, b, Options())

function nnlsobjective(g, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    d = A*x - b
    val = sum(d.^2)/2
    if !(g === nothing)
        At_mul_B!(g, A, d)
    end
    val
end

function precondprepnnls(P, x, mu, a)
    for i = 1:length(x)
        P[i] = 1/(mu/x[i]^2 + a[i])
    end
end

export nnls
