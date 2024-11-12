using Optim, NLSolversBase #hide
import NLSolversBase: clear! #hide

fun(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function fun_grad!(g, x)
g[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
g[2] = 200.0 * (x[2] - x[1]^2)
end

function fun_hess!(h, x)
h[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
h[1, 2] = -400.0 * x[1]
h[2, 1] = -400.0 * x[1]
h[2, 2] = 200.0
end;

x0 = [0.0, 0.0]
df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)

lx = [-0.5, -0.5]; ux = [0.5, 0.5]
dfc = TwiceDifferentiableConstraints(lx, ux)

res = optimize(df, dfc, x0, IPNewton())

ux = fill(Inf, 2)
dfc = TwiceDifferentiableConstraints(lx, ux)

clear!(df)
res = optimize(df, dfc, x0, IPNewton())

lx = fill(-Inf, 2); ux = fill(Inf, 2)
dfc = TwiceDifferentiableConstraints(lx, ux)

clear!(df)
res = optimize(df, dfc, x0, IPNewton())

lx = Float64[]; ux = Float64[]
dfc = TwiceDifferentiableConstraints(lx, ux)

clear!(df)
res = optimize(df, dfc, x0, IPNewton())

con_c!(c, x) = (c[1] = x[1]^2 + x[2]^2; c)
function con_jacobian!(J, x)
    J[1,1] = 2*x[1]
    J[1,2] = 2*x[2]
    J
end
function con_h!(h, x, λ)
    h[1,1] += λ[1]*2
    h[2,2] += λ[1]*2
end;

lx = Float64[]; ux = Float64[]
lc = [-Inf]; uc = [0.5^2]
dfc = TwiceDifferentiableConstraints(con_c!, con_jacobian!, con_h!,
                                     lx, ux, lc, uc)
res = optimize(df, dfc, x0, IPNewton())

lc = [0.1^2]
dfc = TwiceDifferentiableConstraints(con_c!, con_jacobian!, con_h!,
                                     lx, ux, lc, uc)
res = optimize(df, dfc, x0, IPNewton())

function con2_c!(c, x)
    c[1] = x[1]^2 + x[2]^2     ## First constraint
    c[2] = x[2]*sin(x[1])-x[1] ## Second constraint
    c
end
function con2_jacobian!(J, x)
    # First constraint
    J[1,1] = 2*x[1]
    J[1,2] = 2*x[2]
    # Second constraint
    J[2,1] = x[2]*cos(x[1])-1.0
    J[2,2] = sin(x[1])
    J
end
function con2_h!(h, x, λ)
    # First constraint
    h[1,1] += λ[1]*2
    h[2,2] += λ[1]*2
    # Second constraint
    h[1,1] += λ[2]*x[2]*-sin(x[1])
    h[1,2] += λ[2]*cos(x[1])
    # Symmetrize h
    h[2,1]  = h[1,2]
    h
end;

x0 = [0.25, 0.25]
lc = [-Inf, 0.0]; uc = [0.5^2, 0.0]
dfc = TwiceDifferentiableConstraints(con2_c!, con2_jacobian!, con2_h!,
                                     lx, ux, lc, uc)
res = optimize(df, dfc, x0, IPNewton())

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
