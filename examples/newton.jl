load("src/optim.jl")

f = x -> (x - 5) ^ 4
g = x -> 4(x - 5) ^ 3
h = x -> 12(x - 5) ^ 2

newton(f, g, h, 0, 10e-16, 0.1, 0.8)
newton(f, g, h, 0, 10e-32, 0.1, 0.8)
newton(f, g, h, 0, 10e-40, 0.1, 0.8)
