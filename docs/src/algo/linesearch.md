# Line search
## Description

### Available line search algorithms

* `hz_linesearch!` , the default line search algorithm
* `backtracking_linesearch!`
* `interpolating_linesearch!`
* `mt_linesearch!`

The default line search algorithm is taken from the Conjugate Gradient implementation
by Hager and Zhang (HZ).

## Example
## References
W. W. Hager and H. Zhang (2006) "Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent."" ACM Transactions on Mathematical Software 32: 113-137.
Wright, Stephen, and Jorge Nocedal (2006) "Numerical optimization." Springer
