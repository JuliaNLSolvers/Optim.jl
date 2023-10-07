# generic version for gpu support
function maxdiff(x::AbstractArray, y::AbstractArray)
    return mapreduce((a, b) -> abs(a - b), max, x, y)
end

# allocation free version for normal arrays
function maxdiff(x::Array, y::Array)
    res = real(zero(x[1] - y[1]))
    @inbounds for i in 1:length(x)
        delta = abs(x[i] - y[i])
        if delta > res
            res = delta
        end
    end
    return res
end
