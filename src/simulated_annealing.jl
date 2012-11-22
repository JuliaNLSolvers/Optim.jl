using Base

##
#
# simulated_annealing
# Arguments:
# * cost: Function from states to the real numbers. Often called an energy function, but this algorithm works for both positive and negative costs.
# * s0: The initial state of the system.
# * neighbor: Function from states to states. Produces what the Metropolis algorithm would call a proposal.
# * temperature: Function specifying the temperature at time i.
# * iterations: How many iterations of the algorithm should be run? This is the only termination condition.
# * keep_best: Do we return the best state visited or the last state visisted? (Should default to true.)
# * show_trace: Do we show a trace of the system's evolution?
#
##

function simulated_annealing(cost::Function,
                             s0::Any,
                             neighbor::Function,
                             temperature::Function,
                             keep_best::Bool,
                             tolerance::Float64,
                             max_iterations::Int64,
                             show_trace::Bool)
  
  # Set our current state to the specified intial state.
  s = s0
  y = cost(s)
  
  # Set the best state we've seen to the intial state.
  best_s = s0

  i = 0
  if show_trace
    println("Iteration: $(i)")
    println("s = $(s)")
    println("y = $(y)")
    println()
  end

  # We always perform a fixed number of iterations.
  while i < max_iterations
    # Update the iteration counter.
    i = i + 1
    
    # Call temperature to find the proper temperature at time i.
    t = temperature(i)
    
    # Call neighbor to randomly generate a neighbor of our current state.
    s_n = neighbor(s)
    
    # Evaluate the cost function on our current and its neighbor.
    y = cost(s)
    y_n = cost(s_n)
        
    if y_n <= y
      # If the proposed new state is superior, we always move to it.
      s = s_n
    else
      # If the proposed new state is inferior, we move to it with
      # probability p.
      p = exp(- ((y_n - y) / t))
            
      if rand() <= p
        s = s_n
      else
        s = s
      end
    end
    
    # If the new state is the best state we have seen, keep a record of it.
    if cost(s) < cost(best_s)
      best_s = s
    end
    
    # Print out the state of the system.
    if show_trace
      println("Iteration: $(i)")
      println("s = $(s)")
      println("s_n = $(s_n)")
      println("y = $(y)")
      println("y_n = $(y_n)")
      println()
    end
  end
  
  # If specified by the user, we return the best state we've seen.
  # Otherwise, we return the late state we've seen.
  if keep_best
    OptimizationResults("Simulated Annealing", s0, best_s, cost(best_s), max_iterations, false)
  else
    OptimizationResults("Simulated Annealing", s0, s, cost(s), max_iterations, false)
  end
end

##
#
# Off-the-shelf cooling schedules
#
##

# Theoretically, SA will converge if the temperature decreases
# according inversely proportional to the current time.
#
# For this to work, an unknown constant must be used.
# In practice, we use 1 instead of this unknown constant.
function log_temperature(i)
  1 / log(i)
end

# If the temperature is held constant and the cost function is -log(p),
# SA reduces to the Metropolis algorithm for sampling from a distribution.
function constant_temperature(i)
  1
end

function simulated_annealing(cost::Function,
                             s0::Any,
                             neighbor::Function)
  simulated_annealing(cost, s0, neighbor, log_temperature, true, 10e-8, 100_000, false)
end

function simulated_annealing(cost::Function,
                             s0::Vector)
  function neighbor(x)
    map(x_i -> rand_cauchy(x_i, 1), x)
  end
  simulated_annealing(cost, s0, neighbor, log_temperature, true, 10e-8, 100_000, false)
end
