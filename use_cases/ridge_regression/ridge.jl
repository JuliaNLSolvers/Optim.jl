load("src/init.jl")

# Given inputs, ouputs, lambda and current weights,
# Assess the ridge loss function.
# Assume first entry of w is an intercept and do not penalize.

function ridge_error(x, y, l, w)
  (1.0 / 2.0) * sum(map(z -> z^2, y - x * w)) + (l / 2.0) * sum(map(z -> z^2, w[2:length(w)]))
end

ridge_error([1.0 2; 3 4], [1, 2], 1, [1, 1])

function ridge_gradient(x, y, l, w)
	unshift(l * w[2:length(w)], 0) - x' * (y - x * w)
end

ridge_gradient([1.0 1 2; 1 3 4], [1, 2], 1, [1, 1, 1])

function ridge_regression(x, y, l)
	function f(w)
    ridge_error(x, y, l, w)
  end
	
  function g(w)
    ridge_gradient(x, y, l, w)
  end
	
	w0 = zeros(size(x, 2))
	
	solution = gradient_descent(f, g, w0)
	
	solution.minimum
end

x = [1 1 2; 1 3 3; 1 5 6]
y = [1, 2, 2]

w = ridge_regression(x, y, 0.0)
x * w
norm(w)

w = ridge_regression(x, y, 1.0)
x * w
norm(w)

w = ridge_regression(x, y, 10.0)
x * w
norm(w)

w = ridge_regression(x, y, 100.0)
x * w
norm(w)

p = 10

x = hcat([1, 1], vcat([1:p]', [(p + 1):(2p)]'))
y = [1, 2]

w = ridge_regression(x, y, 0)
x * w
norm(w)

w = ridge_regression(x, y, 1)
x * w
norm(w)

w = ridge_regression(x, y, 10)
x * w
norm(w)
