# Simple forward-differencing based on Nocedal and Wright and
# suggestions from Nathaniel Daw.
#
# Will need to implement central-differencing as well. Wrap both.

# Arguments:
# f: A function.
# x: A vector in the domain of f.

function estimate_gradient(f::Function)
  function g(x::Vector)
    # Establish a baseline value of f(x).
    f_x = f(x)
    
    # How far do we move in each direction?
    # See Nodedal and Wright for motivation.
    alpha = sqrt(10e-16)
    
    # What is the dimension of x?
    n = length(x)
    
    # Compute forward differences.
    diff = zeros(n)
    
    # Iterate over each dimension separately.
    for j = 1:n
      dx = zeros(n)
      dx[j] = alpha
      diff[j] = f(x + dx)
    end
    
    # Estimate gradient by dividing out by forward movement.
    (diff - f_x) ./ alpha
  end
  
  # Return a closure.
  g
end
