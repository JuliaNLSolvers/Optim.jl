function nnls(A::AbstractMatrix, b::AbstractVector, ops::Options)
    # Solve the unconstrained problem
    x = A\b
    # Generate an interior point
    @defaults ops xmin=one(eltype(x))
    flag = x .< xmin
    if isa(xmin, AbstractVector)
        x[flag] = xmin[flag]
    else
        x[flag] = xmin
    end
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
        At_mul_B(g, A, d)
    end
    val
end

export nnls
