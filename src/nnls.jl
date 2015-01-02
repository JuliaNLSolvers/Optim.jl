function nnls(A::AbstractMatrix, b::AbstractVector)
    # Set up the preconditioner as the inverse diagonal of the Hessian
    a = sum(A.^2, 1)
    # Create the initial guess (an interior point)
    T = promote_type(eltype(A), eltype(b))
    x = fill(one(T), size(A, 2))
    # Set up constraints
    l = zeros(eltype(x), length(x))
    u = fill(convert(eltype(x), Inf), length(x))
    # Perform the optimization    
    func = (x, g) -> nnlsobjective(x, g, A, b)
    df = DifferentiableFunction(x->func(x,nothing), func, func)
    fminbox(df, x, l, u, precondprep=(P, x, l, u, mu)->precondprepnnls(P, x, mu, a))
end

function nnlsobjective(x::AbstractVector, g, A::AbstractMatrix, b::AbstractVector)
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
