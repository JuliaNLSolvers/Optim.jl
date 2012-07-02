load("src/init.jl")

# Given inputs, ouputs, lambda and current weights,
# Assess the ridge loss function.
# Assume first entry of w is an intercept and do not penalize.

function ridge_error(x, y, l, w)
  (1.0 / 2) * sum(map(x -> x ^ 2, y - x * w)) + (l / 2) * sum(map(x -> x ^ 2, w[2:length(w)]))
end

ridge_error([1.0 2; 3 4], [1, 2], 1, [1, 1])

function ridge_gradient(x, y, l, w)
	unshift(l * w[2:length(w)], 0) - x' * (y - x * w)
end

ridge_gradient([1.0 1 2; 1 3 4], [1, 2], 1, [1, 1, 1])

function ridge_regression(x, y, l)
	f = w -> ridge_error(x, y, l, w)
	
	g = w -> ridge_gradient(x, y, l, w)
	
	w0 = zeros(size(x, 2))
	
	solution = gradient_descent2(f, g, w0, 10e-8, 0.01, 0.8)
	
	solution.minimum
end

x = [1 1 2; 1 3 4]
y = [1, 2]
l = 1

w = ridge_regression(x, y, 0)
x * w

w = ridge_regression(x, y, 1)
x * w

w = ridge_regression(x, y, 10)
x * w

w = ridge_regression(x, y, 100)
x * w

p = 10

x = hcat([1, 1], vcat([1:p]', [(p + 1):(2p)]'))
y = [1, 2]
l = 1

w = ridge_regression(x, y, 0)
x * w

w = ridge_regression(x, y, 1)
x * w

w = ridge_regression(x, y, 10)
x * w
