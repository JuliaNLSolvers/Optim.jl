function maxdiff(x::AbstractArray, y::AbstractArray)
    res = real(zero(x[1] - y[1]))
    @inbounds for i in 1:length(x)
        delta = abs(x[i] - y[i])
        if delta > res
            res = delta
        end
    end
    return res
end
