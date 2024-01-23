using Optim, Random #hide

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

function g!(storage, β)
  calculate_common!(β, last_β)
  for j in 1:m
    storage[j] = s[j]
    for l in 1:m
      storage[j] -= ϵ[j] * f[l] * (H[j,j,l+1] / S[l+1])
    end
  end
end

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

delta_hat = res.minimizer
[delta delta_hat]

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
