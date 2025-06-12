# Simulated Annealing
## Constructor
```julia
SimulatedAnnealing(; neighbor = default_neighbor!,
                     temperature = default_temperature)
```

The constructor takes two keywords:

* `neighbor = a!(x_current, x_proposed)`, a mutating function of the current x, and the proposed x
* `temperature = b(iteration)`, a function of the current iteration that returns a temperature

## Description
Simulated Annealing is a derivative free method for optimization. It is based on
the Metropolis-Hastings algorithm that was originally used to generate samples
from a thermodynamics system, and is often used to generate draws from a posterior
when doing Bayesian inference. As such, it is a probabilistic method for finding
the minimum of a function, often over a quite large domains. For the historical
reasons given above, the algorithm uses terms such as cooling, temperature, and
acceptance probabilities.

As the constructor shows, a simulated annealing implementation is characterized
by a temperature and a neighbor function. The temperature controls how volatile the changes in
minimizer candidates are allowed to be: if `T` is the current temperature, and the objective
values of the current and proposed solutions are `f_current` and `f_proposal`, respectively,
then the probability that the proposed solution will be accepted is
```julia
exp(-(f_proposal - f_current)/T)
```
Note that this implies that the proposed solution is guaranteed to be accepted
if `f_proposal <= f_current`, because this probability becomes larger than 1.
If conversely `f_proposal > f_current`, there is still a chance that it will
be accepted, depending on the temperature, with higher temperatures making
acceptance more likely.

To obtain a new `f_proposal`, we need a neighbor function. A simple neighbor
function adds a standard normal draw to each dimension of `x`
```julia
function neighbor!(x::Array, x_proposal::Array)
    for i in eachindex(x)
        x_proposal[i] = x[i]+randn()
    end
end
```
However, some problems may require custom neighbor functions.

This implementation of Simulated Annealing is a quite simple version of Simulated Annealing
without many bells and whistles. In Optim.jl, we also have the `SAMIN` algorithm implemented.
Consider reading the docstring or documentation page for `SAMIN` to learn about an alternative
Simulated Annealing implementation that additionally allows you to set bounds on the sampling
domain.

## Example

Given a graph adjacency matrix `J`, the [max-cut problem](https://en.wikipedia.org/wiki/Maximum_cut)
may be solved as follows:

```julia
maxcut_objective(x::AbstractVector, J::AbstractMatrix{Bool}) = x' * (J * x)

function maxcut_spinflip!(xcurrent::AbstractVector, xproposed::AbstractVector, p::Real)
    for i in eachindex(xcurrent, xproposed)
        xproposed[i] = (rand() < p ? -1 : 1) * xcurrent[i]
    end
    return xproposed
end

n = size(J, 1)
x0 = rand([-1.0, 1.0], n)    # each entry is ±1
method = SimulatedAnnealing(; neighbor=(xc, xp) -> maxcut_spinflip!(xc, xp, 2/n))
options = Optim.Options(; iterations = 100_000)
results = Optim.optimize(x -> maxcut_objective(x, J), x0, method, options)
```

## References

Kirkpatrick, S., Gelatt Jr, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. Science, 220(4598), 671-680.
