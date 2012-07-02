# http://optlab-server.sce.carleton.ca/POAnimations2007/NonLinear7.html

# Initial points
# N + 1 points of dimension N
# Store in N + 1 by N array
# Compute centroid of non-maximal points
# Iteratively update p_h
# Parameters: a, g, b
# y_l is current minimum
# y_h is current maximum

function centroid(p::Matrix)
  mean(p, 1)
end

function nelder_mead(f::Function,
                     initial_p::Matrix,
                     a::Float64,
                     g::Float64,
                     b::Float64,
                     tolerance::Float64,
                     max_iterations::Int64,
                     show_trace::Bool)
  
  # Center the algorithm around an arbitrary point.
  p = initial_p
  
  # Maintain a record of the value of f() at n points.
  n = size(p, 1)
  y = zeros(n)
  for i = 1:n
    y[i] = f(p[i, :])
  end
  
  # Don't run forever.
  iter = 0
  
  
  while sqrt(var(y) * ((n - 1) / n)) > tolerance && iter < max_iterations
    # Augment the iteration counter.
    iter = iter + 1
    
    # Find p_l and p_h, the minimum and maximum values of f() among p.
    # Always take the first min or max if many exist.
    l = find(y .== min(y))[1]
    p_l = p[l, :]
    y_l = y[l]
    
    h = find(y .== max(y))[1]
    p_h = p[h, :]
    y_h = y[h]
    
    # Remove the maximum valued point from our set.
    non_maximal_p = p[find([1:n] .!= h), :]
    
    # Compute the centroid of the remaining points.
    p_bar = centroid(non_maximal_p)
    
    # For later, prestore function values of all non-maximal points.
    y_bar = zeros(n - 1)
    
    for i = 1:(n - 1)
      y_bar[i] = f(non_maximal_p[i, :])
    end
    
    # Compute a reflection.
    p_star = (1 + a) * p_bar - a * p_h
    
    y_star = f(p_star)
    
    if y_star < y_l
      # Compute an expansion.
      p_star_star = g * p_star + (1 - g) * p_bar
      y_star_star = f(p_star_star)
      
      if y_star_star < y_l
        p_h = p_star_star
        p[h, :] = p_h
      else
        p_h = p_star
        p[h, :] = p_h
      end
    else
      if all(y_star .> y_bar)
        if y_star > y_h
          1 # Do a NO-OP.
        else
          p_h = p_star
          p[h, :] = p_h
        end
        
        # Compute a contraction.
        p_star_star = b * p_h + (1 - b) * p_bar
        y_star_star = f(p_star_star)
        
        if y_star_star > y_h
          for i = 1:n
            p[i, :] = (p[i, :] + p_l) / 2
          end
        else
          p_h = p_star_star
          p[h, :] = p_h
        end
      else
        p_h = p_star
        p[h, :] = p_h
      end
    end
    
    # Recompute y's to assess convergence.
    y = zeros(n, 1)
    for i = 1:n
      y[i] = f(p[i, :])
    end
	  
  	if show_trace
  	  println(p)
  	  println()
  	end
  end
  
  (centroid(p), f(centroid(p)), iter)
end
