# http://optlab-server.sce.carleton.ca/POAnimations2007/NonLinear7.html

# Initial points
# N + 1 points of dimension N
# Store in N + 1 by N array
# Compute centroid of non-maximal points
# Iteratively update p_h
# Parameters: a, g, b
# y_l is current minimum
# y_h is current maximum

function centroid(x)
  mean(x, 1)
end

function nelder_mead(f, initial_p, a, g, b, tolerance, max_iterations)
  p = initial_p
  
  y = zeros(size(p, 1), 1)
  for i = 1:size(p, 1)
    y[i, 1] = f(p[i, :])
  end
  
  iter = 0
  while std(y) / sqrt(length(y)) > tolerance && iter < max_iterations
    iter = iter + 1
        
    # Find p_l and p_h
    # Always take the first min or max if many exist.
    l = find(y == min(y))[1]
    p_l = p[l, :]
    y_l = y[l, 1]
    
    h = find(y == max(y))[1]
    p_h = p[h, :]
    y_h = y[h, 1]
    
    non_maximal_p = p[find([1:size(p, 1)] != h), :]
    
    p_bar = centroid(non_maximal_p)
    
    # For later, prestore function values of all non-maximal points.
    y_bar = zeros(size(non_maximal_p, 1), 1)
    
    for i = 1:size(non_maximal_p, 1)
      y_bar[i, 1] = f(non_maximal_p[i, :])
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
      if all(y_star > y_bar)
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
          for i = 1:size(p, 1)
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
    y = zeros(size(p, 1), 1)
    for i = 1:size(p, 1)
      y[i, 1] = f(p[i, :])
    end    
  end
  
  (centroid(p), f(centroid(p)), iter)
end
