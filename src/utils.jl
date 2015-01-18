# Dot product of two "vectors", even if they don't have a vector shape
_dot(x::Vector, y::Vector) = dot(x, y)
function _dot(x::Array, y::Array)
    # TODO: Check length here?
    if length(x) != length(y)
        throw(DomainError())
    end
    d = x[1] * y[1]
    for i = 2:length(x)
        d += x[i] * y[i]
    end
    return d
end

# Vector-norm-squared, even if it doesn't have a vector shape
norm2(x::Array) = _dot(x, x)
