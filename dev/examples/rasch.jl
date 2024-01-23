# # Conditional Maximum Likelihood for the Rasch Model
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [``rasch.ipynb``](@__NBVIEWER_ROOT_URL__examples/generated/rasch.ipynb)
#-
using Optim, Random #hide
#
# The Rasch model is used in psychometrics as a model for
# assessment data such as student responses to a standardized
# test. Let $X_{pi}$ be the response accuracy of student $p$
# to item $i$ where $X_{pi}=1$ if the item was answered correctly
# and $X_{pi}=0$ otherwise for $p=1,\ldots,n$ and $i=1,\ldots,m$. 
# The model for this accuracy is 
# ```math
#   P(\mathbf{X}_{p}=\mathbf{x}_{p}|\xi_p, \mathbf\epsilon) = \prod_{i=1}^m \dfrac{(\xi_p \epsilon_j)^{x_{pi}}}{1 + \xi_p\epsilon_i}
# ```
# where $\xi_p > 0$ the latent ability of person $p$ and $\epsilon_i > 0$ 
# is the difficulty of item $i$. 

# We simulate data from this model:

Random.seed!(123)
n = 1000
m = 5
theta = randn(n)
delta = randn(m)
r = zeros(n)
s = zeros(m)
  
for i in 1:n
  p = exp.(theta[i] .- delta) ./ (1.0 .+ exp.(theta[i] .- delta))  
  for j in 1:m
    if rand() < p[j] ##correct
      r[i] += 1
      s[j] += 1
    end
  end
end
f = [sum(r.==j) for j in 1:m];

# Since the number of parameters increases 
# with sample size standard maximum likelihood will not provide us 
# consistent estimates. Instead we consider the conditional likelihood. 
# It can be shown that the Rasch model is an exponential family model and
# that the sum score $r_p = \sum_{i} x_{pi}$ is the sufficient statistic for
# $\xi_p$. If we condition on the sum score we should be able to eliminate
# $\xi_p$. Indeed, with a bit of algebra we can show
# ```math
# P(\mathbf{X}_p = \mathbf{x}_p | r_p, \mathbf\epsilon) = \dfrac{\prod_{i=1}^m \epsilon_i^{x{ij}}}{\gamma_{r_i}(\mathbf\epsilon)}
# ```
# where $\gamma_r(\mathbf\epsilon)$ is the elementary symmetric function of order $r$
# ```math
# \gamma_r(\mathbf\epsilon) = \sum_{\mathbf{y} : \mathbf{1}^\intercal \mathbf{y} = r} \prod_{j=1}^m \epsilon_j^{y_j}
# ```
# where the sum is over all possible answer configurations that give a sum
# score of $r$. Algorithms to efficiently compute $\gamma$ and its 
# derivatives are available in the literature (see eg Baker (1996) for a review
# and Biscarri (2018) for a more modern approach)

function esf_sum!(S::AbstractArray{T,1}, x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  fill!(S,zero(T))
  S[1] = one(T)
  @inbounds for col in 1:n
    for r in 1:col
      row = col - r + 1 
      S[row+1] = S[row+1] + x[col] * S[row]
    end
  end
end

function esf_ext!(S::AbstractArray{T,1}, H::AbstractArray{T,3}, x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  esf_sum!(S, x)
  H[:,:,1] .= zero(T)
  H[:,:,2] .= one(T)

  @inbounds for i in 3:n+1
    for j in 1:n
      H[j,j,i] = S[i-1] - x[j] * H[j,j,i-1]
      for k in j+1:n
        H[k,j,i] = S[i-1] - ((x[j]+x[k])*H[k,j,i-1] + x[j]*x[k]*H[k,j,i-2])
        H[j,k,i] = H[k,j,i]
      end
    end
  end
end

# The objective function we want to minimize is the negative log conditional
# likelihood
# ```math
# \begin{aligned}
# \log{L_C(\mathbf\epsilon|\mathbf{r})} &= \sum_{p=1}^n \sum_{i=1}^m x_{pi} \log{\epsilon_i} - \log{\gamma_{r_p}(\mathbf\epsilon)}\\
#   &= \sum_{i=1}^m s_i \log{\epsilon_i} - \sum_{r=1}^m f_r \log{\gamma_r(\mathbf\epsilon)}
# \end{aligned}
# ```
ϵ = ones(Float64, m)
β0 = zeros(Float64, m)
last_β = fill(NaN, m)
S = zeros(Float64, m+1)
H = zeros(Float64, m, m, m+1)

function calculate_common!(x, last_x)
  if x != last_x
    copyto!(last_x, x)
    ϵ .= exp.(-x)
    esf_ext!(S, H, ϵ)
  end
end
function neglogLC(β)
  calculate_common!(β, last_β)
  return -s'log.(ϵ) + f'log.(S[2:end])
end

# Parameter estimation is usually performed with respect to the unconstrained parameter 
# $\beta_i = -\log{\epsilon_i}$. Taking the derivative with respect to $\beta_i$ 
# (and applying the chain rule) one obtains 
# ```math
#   \dfrac{\partial\log L_C(\mathbf\epsilon|\mathbf{r})}{\partial \beta_i} = -s_i + \epsilon_i\sum_{r=1}^m \dfrac{f_r \gamma_{r-1}^{(j)}}{\gamma_r} 
# ```
# where $\gamma_{r-1}^{(i)} = \partial \gamma_{r}(\mathbf\epsilon)/\partial\epsilon_i$. 

function g!(storage, β)
  calculate_common!(β, last_β)
  for j in 1:m
    storage[j] = s[j]
    for l in 1:m
      storage[j] -= ϵ[j] * f[l] * (H[j,j,l+1] / S[l+1]) 
    end
  end
end

# Similarly the Hessian matrix can be computed
# ```math
#   \dfrac{\partial^2 \log L_C(\mathbf\epsilon|\mathbf{r})}{\partial \beta_i\partial\beta_j} = \begin{cases} \displaystyle  -\epsilon_i \sum_{r=1}^m \dfrac{f_r\gamma_{r-1}^{(i)}}{\gamma_r}\left(1 - \dfrac{\gamma_{r-1}^{(i)}}{\gamma_r}\right) & \text{if $i=j$}\\
#     \displaystyle -\epsilon_i\epsilon_j\sum_{r=1}^m \dfrac{f_r \gamma_{r-2}^{(i,j)}}{\gamma_r} - \dfrac{f_r\gamma_{r-1}^{(i)}\gamma_{r-1}^{(j)}}{\gamma_r^2} &\text{if $i\neq j$}
#    \end{cases}
# ```
# where $\gamma_{r-2}^{(i,j)} = \partial^2 \gamma_{r}(\mathbf\epsilon)/\partial\epsilon_i\partial\epsilon_j$.

function h!(storage, β)
  calculate_common!(β, last_β)
  for j in 1:m
    for k in 1:m
      storage[k,j] = 0.0
      for l in 1:m
        if j == k
          storage[j,j] += f[l] * (ϵ[j]*H[j,j,l+1] / S[l+1]) * 
            (1 - ϵ[j]*H[j,j,l+1] / S[l+1])
        elseif k > j
          storage[k,j] += ϵ[j] * ϵ[k] * f[l] * 
            ((H[k,j,l] / S[l+1]) - (H[j,j,l+1] * H[k,k,l+1]) / S[l+1] ^ 2)
        else #k < j
          storage[k,j] += ϵ[j] * ϵ[k] * f[l] * 
            ((H[j,k,l] / S[l+1]) - (H[j,j,l+1] * H[k,k,l+1]) / S[l+1] ^ 2)
        end
      end
    end
  end
end

# The estimates of the item parameters are then obtained via standard optimization 
# algorithms (either Newton-Raphson or L-BFGS). One last issue is that the model is 
# not identifiable (multiplying the $\xi_p$ by a constant and dividing the $\epsilon_i$ 
# by the same constant results in the same likelihood). Therefore some kind of constraint 
# must be imposed when estimating the parameters. Typically either $\epsilon_1 = 0$ or 
# $\prod_{i=1}^m \epsilon_i = 1$ (which is equivalent to $\sum_{i=1}^m \beta_i = 0$).

con_c!(c, x) = (c[1] = sum(x); c)
function con_jacobian!(J, x)
  J[1,:] .= ones(length(x))
end
function con_h!(h, x, λ)
    for i in 1:size(h)[1]
        for j in 1:size(h)[2]
            h[i,j] += (i == j) ? λ[1] : 0.0
        end
    end
end
lx = Float64[]; ux = Float64[]
lc = [0.0]; uc = [0.0]
df = TwiceDifferentiable(neglogLC, g!, h!, β0)
dfc = TwiceDifferentiableConstraints(con_c!, con_jacobian!, con_h!, lx, ux, lc, uc)
res = optimize(df, dfc, β0, IPNewton())

# Compare the estimate to the truth
delta_hat = res.minimizer
[delta delta_hat]
