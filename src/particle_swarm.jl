using Distributions

immutable ParticleSwarm <: Optimizer end

function particle_swarm{T}(cost_function::Function,
                           initial_x::Vector{T},
                           xmin::Vector{T},
                           xmax::Vector{T};
                           nParticles::Int = length(initial_x)
                           maxIter::Int = 200,
                           showTrace::Bool = false)


  nDim = length(initial_x)
  c1 = 2.0
  c2 = 2.0
  w = 1.0

  X = Array{T}(nDim, nParticles)
  V = Array{T}(nDim, nParticles)
  X_best = Array{T}(nDim, nParticles)
  dx = zeros(T, nDim)
  score = zeros(T, nParticles)
  best_point = zeros(T, nDim)
  best_score = zeros(T, nParticles)
  xlearn = zeros(T, nDim)

  iteration = 0
  f_calls = 0
  current_state = 0
  best_score_global = 0.0

  for i in 1:nParticles
    for j in 1:nDim
      if i == 1
        if abs(initial_x[i]) > 0.0
          dx[j] = abs(initial_x[i])
        else
          dx[j] = 1.0
        end
      end
      X[j,i] = initial_x[j] + dx[j] * rand()
      X_best[j,i] = X[j,i]
      V[i,j] = abs(X[i,j]) * (rand() * 2.0 - 1.0)
    end
  end

  for j in 1:nDim
    X[j,1] = initial_x[j]
    X_best[j,1] = initial_x[j]
  end
  tr = OptimizationTrace()
  doLimitSearchSpace = false
  if length(xmin) >= 1
    doLimitSearchSpace = true
  end

  while (iteration <= nIterations)
    if doLimitSearchSpace
      limit_X!(X, xmin, xmax, nParticles, nDim)
    end
    compute_cost!(cost_function, nParticles, X, score)
    f_calls += nParticles

    if iteration == 0
      for i=1:nParticles
        best_score[i] = score[i]
      end
      best_score_global = minimum(score)
    end
    best_score_global = housekeeping!(score, best_score, X, X_best, best_point,
                                best_score_global, nParticles)

    # Elitist Learning:
    # find a new solution named 'xlearn' which is the current best
    # solution with one randomly picked variable being modified.
    # Replace the current worst solution in X with xlearn
    # if xlearn presents the new best solution.
    # In all other cases discard xlearn.
    # This helps jumping out of local minima.
    worst_score, iWorst = findmax(score)
    for k=1:nDim
      xlearn[k] = best_point[k]
    end
    random_index = rand(1:nDim)
    random_value = randn()
    sigma_learn = 1 - (1 - 0.1) * iteration / nIterations
    dist = Normal(0, sigma_learn)
    if doLimitSearchSpace
      xlearn[random_index] = xlearn[random_index] + (xmax[random_index] - xmin[random_index]) / 3.0 * rand(dist)
    else
      xlearn[random_index] = xlearn[random_index] + xlearn[random_index] * rand(dist)
    end

    if doLimitSearchSpace
      if xlearn[random_index] < xmin[random_index]
        xlearn[random_index] = xmin[random_index]
      elseif xlearn[random_index] > xmax[random_index]
        xlearn[random_index] = xmax[random_index]
      end
    end

    score_learn = cost_function(xlearn)
    f_calls += 1
    if score_learn < best_score_global
      best_score_global = score_learn * 1.0
      for j = 1:nDim
        X_best[j,iWorst] = xlearn[j]
        X[j,iWorst] = xlearn[j]
        best_point[j] = xlearn[j]
      end
      score[iWorst] = score_learn
      best_score[iWorst] = score_learn
    end

    current_state, f = get_swarm_state(X, score, best_point, current_state)
    w, c1, c2 = update_swarm_params!(c1, c2, w, current_state, f)
    update_swarm!(X, X_best, best_point, nDim, nParticles, V, w, c1, c2)

    iteration += 1
  end

  f_converged = true
  ftol = 1e-2

  return MultivariateOptimizationResults("Particle-Swarm",
                                         initial_x,
                                         best_point,
                                         best_score_global,
                                         iteration,
                                         iteration == nIterations,
                                         false,
                                         NaN,
                                         f_converged,
                                         ftol,
                                         false,
                                         NaN,
                                         tr,
                                         f_calls,
                                         0)
end

function update_swarm!(X, X_best, best_point, nDim, nParticles, V,
                       w, c1, c2)
  for i=1:nParticles
    for j=1:nDim
      r1 = rand()
      r2 = rand()
      vx = X_best[j,i] - X[j,i]
      vg = best_point[j] - X[j,i]
      V[j,i] = V[j,i]*w + c1*r1*vx + c2*r2*vg
      X[j,i] = X[j,i] + V[j,i]
    end
  end
end

function get_mu_1(f)
  if 0 <= f <= 0.4
    return 0.0
  elseif 0.4 < f <= 0.6
    return 5. * f - 2.0
  elseif 0.6 < f <= 0.7
    return 1.0
  elseif 0.7 < f <= 0.8
    return -10. * f + 8.0
  else
    return 0.0
  end
end

function get_mu_2(f)
  if 0 <= f <= 0.2
    return 0.0
  elseif 0.2 < f <= 0.3
    return 10. * f - 2.0
  elseif 0.3 < f <= 0.4
    return 1.0
  elseif 0.4 < f <= 0.6
    return -5. * f + 3.0
  else
    return 0.0
  end
end

function get_mu_3(f)
  if 0 <= f <= 0.1
    return 1.0
  elseif 0.1 < f <= 0.3
    return -5. * f + 1.5
  else
    return 0.0
  end
end

function get_mu_4(f)
  if 0 <= f <= 0.7
    return 0.0
  elseif 0.7 < f <= 0.9
    return 5. * f - 3.5
  else
    return 1.0
  end
end

function get_swarm_state(X, score, best_point, previous_state)
  nDim, nParticles = size(X)
  fBest, iBest = findmin(score)
  d = zeros(Float64, nParticles)
  for i = 1:nParticles
    dd = 0.0
    for k = 1:nParticles
      for dim =1:nDim
        @inbounds ddd = (X[dim,i] - X[dim, k])
        dd += ddd*ddd
      end
    end
    d[i] = sqrt(dd)
  end
  dg = d[iBest]
  dmin = minimum(d)
  dmax = maximum(d)

  f = (dg - dmin) / (dmax - dmin)
  mu = zeros(Float64, 4)
  mu[1] = get_mu_1(f)
  mu[2] = get_mu_2(f)
  mu[3] = get_mu_3(f)
  mu[4] = get_mu_4(f)
  best_mu, i_best_mu = findmax(mu)
  current_state = 0

  if previous_state == 0
    current_state = i_best_mu
  elseif previous_state == 1
    if mu[1] > 0
      current_state = 1
    else
      if mu[2] > 0
        current_state = 2
      elseif mu[4] > 0
        current_state = 4
      else
        current_state = 3
      end
    end
  elseif previous_state == 2
    if mu[2] > 0
      current_state = 2
    else
      if mu[3] > 0
        current_state = 3
      elseif mu[1] > 0
        current_state = 1
      else
        current_state = 4
      end
    end
  elseif previous_state == 3
    if mu[3] > 0
      current_state = 3
    else
      if mu[4] > 0
        current_state = 4
      elseif mu[2] > 0
        current_state = 2
      else
        current_state = 1
      end
    end
  elseif previous_state == 4
    if mu[4] > 0
      current_state = 4
    else
      if mu[1] > 0
        current_state = 1
      elseif mu[2] > 0
        current_state = 2
      else
        current_state = 3
      end
    end
  end

  return current_state, f
end

function update_swarm_params!(c1, c2, w, current_state, f)

  delta_c1 = 0.05 + rand()/20.
  delta_c2 = 0.05 + rand()/20.

  if current_state == 1
    c1 += delta_c1
    c2 -= delta_c2
  elseif current_state == 2
    c1 += delta_c1/2.
    c2 -= delta_c2/2.
  elseif current_state == 3
    c1 += delta_c1/2.
    c2 += delta_c2/2.
  elseif current_state == 4
    c1 -= delta_c1
    c2 -= delta_c2
  end

  if c1 < 1.5
    c1 = 1.5
  elseif c1 > 2.5
    c1 = 2.5
  end

  if c2 < 1.5
    c2 = 2.5
  elseif c2 > 2.5
    c2 = 2.5
  end

  if c1 + c2 > 4.0
    c_total = c1+c2
    c1 = c1/c_total*4
    c2 = c2/c_total*4
  end

  w = 1 / (1 + 1.5 * exp(-2.6*f))
  return w, c1, c2
end

#function update!(score, best_score, X, X_best, best_point,
function housekeeping!(score, best_score, X, X_best, best_point,
                best_score_global, nParticles)
  nDim = size(X,1)
  for i=1:nParticles
    if score[i] <= best_score[i]
      best_score[i] = score[i]
      for k=1:nDim
        X_best[k,i] = X[k,i]
      end
      if score[i] <= best_score_global
        for k=1:nDim
        	best_point[k] = X[k,i]
        end
      	best_score_global = score[i]
      end
    end
  end
  return best_score_global
end

function limit_X!(X, xmin, xmax, nParticles, nDim)
  # limit X values to boundaries
  for i=1:nParticles
    for j=1:nDim
      if X[j, i] < xmin[j]
      	X[j, i] = xmin[j]
      elseif X[j, i] > xmax[j]
      	X[j, i] = xmax[j]
      end
    end
  end
  nothing
end

function compute_cost!(cost_function::Function,
                       nParticles::Int,
                       X::Matrix,
                       score::Vector)

  for i=1:nParticles
    score[i] = cost_function(X[:,i])
  end
  nothing
end
