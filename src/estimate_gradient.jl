# Simple forward-differencing and central-differencing
# based on Nocedal and Wright and
# suggestions from Nathaniel Daw. 

# Arguments:
# f: A function.
# x: A vector in the domain of f.

# Forward-differencing
function estimate_gradient(f::Function)
  # Construct and return a closure.
  function g(x::Vector)    
    # How far do we move in each direction?
    # See Nodedal and Wright for motivation.
    epsilon = sqrt(eps())
    
    # What is the dimension of x?
    n = length(x)
    
    # Compute forward differences.
    diff = zeros(n)
    
    # Establish a baseline value of f(x).
    f_x = f(x)
    
    # Iterate over each dimension of the gradient separately.
    for j = 1:n
      dx = zeros(n)
      dx[j] = epsilon
      diff[j] = f(x + dx) - f_x
      diff[j] /= epsilon
    end
    
    # Return the estimated gradient.
    diff
  end
  g
end

# Central-differencing.
function estimate_gradient2(f::Function)
  # Construct and return a closure.
  function g(x::Vector)    
    # How far do we move in each direction?
    # See Nodedal and Wright for motivation.
    epsilon = sqrt(eps())
    
    # What is the dimension of x?
    n = length(x)
    
    # Compute forward differences.
    diff = zeros(n)
    
    # Iterate over each dimension of the gradient separately.
    for j = 1:n
      dx = zeros(n)
      dx[j] = epsilon
      diff[j] = f(x + dx) - f(x - dx)
      diff[j] /= 2 * epsilon
    end
    
    # Return the estimated gradient.
    diff
  end
  g
end

# Forward-differencing Jacobian
function estimate_jacobian(f::Function)
  # Construct and return a closure.
  function g(x::Vector)    
    # How far do we move in each direction?
    # See Nodedal and Wright for motivation.
    epsilon = sqrt(eps())
    
    # What is the dimension of x?
    n = length(x)
    
    # Establish a baseline value of f(x).
    f_x = f(x)

	# initialize Jacobian matrix
	J = zeros(length(f_x), n)
    
    # Iterate over each dimension of the gradient separately.
    for j = 1:n
      dx = zeros(n)
      dx[j] = epsilon
      J[:,j] = (f(x + dx) - f_x) / epsilon
    end
    
    # Return the estimated Jacobian.
    J
  end
  g
end