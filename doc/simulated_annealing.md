# simulated_annealing()

* Arguments:
  * cost: Function from states to the real numbers. Often called an energy function, but this algorithm works for both positive and negative costs.
  * s0: The initial state of the system.
  * neighbor: Function from states to states. Produces what the Metropolis algorithm would call a proposal.
  * temperature: Function specifying the temperature at time i.
  * iterations: How many iterations of the algorithm should be run? This is the only termination condition.
  * keep_best: Do we return the best state visited or the last state visisted? (Should default to true.)
  * trace: Do we show a trace of the system's evolution?
* Returns:
  * solution: A purported minimum of the cost function, which will be of the same type as s0.
