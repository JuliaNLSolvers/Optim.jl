using Plots
include("0_setup.jl")
include("1_model.jl")
const srandseed = 3337
srand(srandseed)
pgfplots()
## Sieve approximated Bellman solutions
# Number of basis functions
Ks = [1, 4]
pp=plot(;grid=false)

K = 70
R0 = K
deg = 1
B0, X0 = Bases.BX(Bases.Chebyshev, K, R0, x_low, x_high, deg)
# Create utility
U0 = [U.(X0, d) for d = 1:U.nD]
R1 = 64
E, W = Bases.quads(D, R1)

# Generate tomorrow's grid based on X0 and E
X1 = Bases.X1grid(tx1, X0, E, U.nD)

# Solve Sieve IV
sivks = SieveIV.SIVKS(X0, X1, B0, W)
res_norms = SieveIV.solve(sivks, U0, β, rand(X0))

plot!(X0, sivks.V, c = :grey, lw = 3, ls=:solid,labels="limit solution")
tols_ev = []
tols_iv = []
Ks
for (K, lw) in zip(Ks, [1, 2])
deg = 1
B0, X0 = Bases.BX(Bases.Chebyshev, K, R0, x_low, x_high, deg)

# Create utility
U0 = [U.(X0, d) for d = 1:U.nD]
R1 = 64
E, W = Bases.quads(D, R1)

# Generate tomorrow's grid based on X0 and E
X1 = Bases.X1grid(tx1, X0, E, U.nD)

# Solve Sieve EV
U1 = [U.(X1[d1], d) for d1 = 1:U.nD, d = 1:U.nD]
sev = SieveEV.SEV(X0, X1, B0, W)
res_ev = SieveEV.solve(sev, U0, U1, β)
# Solve Sieve IV
sivks = SieveIV.SIVKS(X0, X1, B0, W)
res_iv = SieveIV.solve(sivks, U0, β)

push!(tols_ev, res_ev)
push!(tols_iv, res_iv)
plot!(X0, sivks.V, c = :black, lw=lw,ls=:dot,labels="Sieve IV,  K=$K, L = $(round(Bases.γP(X0, B0), 2)) ")
plot!(X0, sev.V, c = :black, lw=lw,ls=:dash,labels="Sieve EV, K=$K, L = $(round(Bases.γP(X0, B0),2)) ")
end
pp

xlabel!("mileage")
ylabel!("integrated value function(x)")

ylims!(-24,0)

savefig("/home/pkm/Dropbox/0Projected Bellman/paper/2017_patrick_JoE/fig/contractionorquality.tex")
