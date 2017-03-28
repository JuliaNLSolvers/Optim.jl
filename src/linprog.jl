
if Pkg.installed("Clp") != nothing
    @eval using Clp
    lpsolver = Clp
else
    lpsolver = nothing
end

type LinprogSolution
    status
    objval
    sol
    attrs
end

typealias InputVector{T<:Real} Union(Vector{T},Real)

function expandvec(x,len::Integer)
    if isa(x,Vector)
        if length(x) != len
            error("Input size mismatch. Expected vector of length $len but got $(length(x))")
        end
        return x
    else
        return fill(x,len)
    end
end



function linprog(c::InputVector, A::AbstractMatrix, rowlb::InputVector, rowub::InputVector, lb::InputVector, ub::InputVector)
    if lpsolver == nothing
        error("No LP solver installed. Please run Pkg.add(\"Clp\") and reload Optim")
    end
    m = lpsolver.model()
    nrow,ncol = size(A)

    c = expandvec(c, ncol)
    rowlbtmp = expandvec(rowlb, nrow)
    rowubtmp = expandvec(rowub, nrow)
    lb = expandvec(lb, ncol)
    ub = expandvec(ub, ncol)
    
    # rowlb is allowed to be vector of senses
    if eltype(rowlbtmp) == Char
        realtype = eltype(rowubtmp)
        sense = rowlbtmp
        rhs = rowubtmp
        @assert realtype <: Real
        rowlb = Array(realtype, nrow)
        rowub = Array(realtype, nrow)
        for i in 1:nrow
            if sense[i] == '<'
                rowlb[i] = typemin(realtype)
                rowub[i] = rhs[i]
            elseif sense[i] == '>'
                rowlb[i] = rhs[i]
                rowub[i] = typemax(realtype)
            elseif sense[i] == '='
                rowlb[i] = rhs[i]
                rowub[i] = rhs[i]
            else
                error("Unrecognized sense '$(sense[i])'")
            end
        end
    else
        rowlb = rowlbtmp
        rowub = rowubtmp
    end

    lpsolver.loadproblem(m, A, lb, ub, c, rowlb, rowub)
    # optimize is masked by Optim's optimize
    # maybe there's a better way
    lpsolver.optimize(m)
    status = lpsolver.status(m)
    if status == :Optimal
        return LinprogSolution(status, lpsolver.getobjval(m), lpsolver.getsolution(m), Dict())
    else
        return LinprogSolution(status, nothing, [], Dict())
    end
end

linprog(c,A,rowlb,rowub) = linprog(c,A,rowlb,rowub,0,Inf)

export linprog


