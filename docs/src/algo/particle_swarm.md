# Particle Swarm
## Constructor
```julia
ParticleSwarm(; lower = [],
                upper = [],
                n_particles = 0)
```

The constructor takes three keywords:

* `lower = []`, a vector of lower bounds, unbounded below if empty or `Inf`'s
* `upper = []`, a vector of upper bounds, unbounded above if empty or `Inf`'s
* `n_particles = 0`, number of particles in the swarm, defaults to least three

## Description
The Particle Swarm implementation in Optim.jl is the so-called Adaptive Particle
Swarm algorithm in [1]. It attempts to improve global coverage and convergence by
switching between four evolutionary states: exploration, exploitation, convergence,
and jumping out. In the jumping out state it intentially tries to take the best
particle and move it away from its (potentially and probably) local optimum, to
improve the ability to find a global optimum. Of course, this comes a the cost
of slower convergence, but hopefully converges to the global optimum as a result.

## References
[1] Zhan, Zhang, and Chung. Adaptive particle swarm optimization, IEEE Transactions on Systems, Man, and Cybernetics, Part B: CyberneticsVolume 39, Issue 6, 2009, Pages 1362-1381 (2009)
