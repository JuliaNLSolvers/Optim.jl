# Simulated Annealing
## Constructor
```julia
SimulatedAnnealing(; neighbor! = default_neighbor!,
                    T = default_temperature,
                    p = kirkpatrick)
```

The constructor takes three keywords:

* `neighbor! = a!(x_proposed, x_current)`, a mutating function of the current x, and the proposed x
* `T = b(iteration)`, a function of the current iteration that returns a temperature
* `p = c(f_proposal, f_current, T)`, a function of the current temperature, current function value and proposed function value that returns an acceptance probability

## Description
Simulated Annealing is a derivative free method for optimization. It is based on
the Metropolis-Hastings algorithm that was originally used to generate samples
from a thermodynamics system, and is often used to generate draws from a posterior
when doing Bayesian inference. As such, it is a probabilistic method for finding
the minimum of a function, often over a quite large domains. For the historical
reasons given above, the algorithm uses terms such as cooling, temperature, and
acceptance probabilities.

As the constructor shows, a simulated annealing implementation is characterized
by a temperature, a neighbor function, and
an acceptance probability. The temperature controls how volatile the changes in
minimizer candidates are allowed to be, as it enters the acceptance probability.
For example, the original Kirkpatrick et al. acceptance probability function can be written
as follows
```julia
p(f_proposal, f_current, T) = exp(-(f_proposal - f_current)/T)
```
A high temperature makes it more likely that a draw is accepted, by pushing acceptance
probability to 1. As in the Metropolis-Hastings
algorithm, we always accept a smaller function value, but we also sometimes accept a
larger value. As the temperature decreases, we're more and more likely to only accept
candidate `x`'s that lowers the function value. To obtain a new `f_proposal`, we need
a neighbor function. A simple neighbor function adds a standard normal draw to each
dimension of `x`
```julia
function neighbor!(x_proposal::Array, x::Array)
    for i in eachindex(x)
        x_proposal[i] = x[i]+randn()
    end
end
```
As we see, it is not really possible
to disentangle the role of the different components of the algorithm. For example, both the
functional form of the acceptance function, the temperature and (indirectly) the neighbor
function determine if the next draw of `x` is accepted or not.

The current implementation of Simulated Annealing is very rough.  It lacks quite
a few features which are normally part of a proper SA implementation.
A better implementation is under way, see [this issue](https://github.com/JuliaOpt/Optim.jl/issues/200).

## Example

## References
