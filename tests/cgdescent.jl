using Optim

gtol = 1e-5

# Quadratic objective function
function quadratic(g, x, A, b)
    calc_grad = !(g === nothing)
    v = x'*A*x/2 + b'*x
    if calc_grad
        A_mul_B(g, A, x)
        for i = 1:numel(g)
            g[i] += b[i]
        end
    end
    return v[1]
end


# Matlab objective function in objfun.m
# Minimum value is 0, which is tricky
function mobjfun(g, x)
    calc_grad = !(g === nothing)
    x1 = x[1]
    x2 = x[2]
    ex1 = exp(x1)
    v = ex1*(4x1^2 + 2x2^2 + 4x1*x2 + 2x2 + 1)
    if calc_grad
        g[1] = v + ex1*(8x1 + 4x2)
        g[2] = ex1*(4x2 + 4x1 + 2)
    end
    return v
end

# driver1 problem in CG_DESCENT
function driver1{T}(g, x::Array{T})
    calc_grad = !(g === nothing)
    f = zero(T)
    n = length(x)
    for i = 1:n
        t = sqrt(convert(T, i))
        ex = exp(x[i])
        f += ex - t*x[i]
        if calc_grad
            g[i] = ex-t
        end
    end
    return f
end

# Quadratic
N = 8
A = randn(N,N)
A = A'*A
b = randn(N)
x0 = randn(N)
# dispflags = OptimizeMod.FINAL | OptimizeMod.ITER# | OptimizeMod.LINESEARCH | OptimizeMod.BRACKET | OptimizeMod.BISECT
# ops = @options display=dispflags
func = (g, x) -> quadratic(g, x, A, b)
x, fval, fcount, converged = cgdescent(func, x0) #, ops)
@assert converged
@assert all(abs(A*x + b) .< gtol)

l = fill(-2.0, N)
u = fill(2.0, N)
x0 = rand(N)-0.5
x, fval, fcount, converged = fminbox(func, x0, l, u) #, ops)
@assert converged

# Matlab objective function
x0 = Float64[-1,1]
g = Array(Float64, 2)
mobjfun(g, x0)
fun = x->mobjfun(nothing, x)
d = derivative_numer(fun, x0, 1)
@assert abs(d-g[1]) < 1e-8
d = derivative_numer(fun, x0, 2)
@assert abs(d-g[2]) < 1e-8
func = (g, x) -> mobjfun(g, x)
x, fval, fcount, converged = cgdescent(func, x0) #, ops)
@assert converged

# cgdescent driver1
n = 100
x0 = ones(n)
func = (g,x) -> driver1(g, x)
x, fval, fcount, converged = cgdescent(func, x0) #, ops)
