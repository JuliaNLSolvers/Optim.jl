## return a uniform random sample from the interval (a, b)
function rand_uniform(a, b)
    a + rand()*(b - a)
end

## return a random sample from a normal (Gaussian) distribution
function rand_normal(mean, stdev)
    if stdev <= 0.0
        error("standard deviation must be positive")
    end
    u1 = rand()
    u2 = rand()
    r = sqrt( -2.0*log(u1) )
    theta = 2.0*pi*u2
    mean + stdev*r*sin(theta)
end

## return a random sample from an exponential distribution
function rand_exponential(mean)
    if mean <= 0.0
        error("mean must be positive")
    end
    -mean*log(rand())
end

## return a random sample from a gamma distribution
function rand_gamma(shape, scale)
    if shape <= 0.0
        error("Shape parameter must be positive")
    end
    if scale <= 0.0
        error("Scale parameter must be positive")
    end
    
    ## Implementation based on "A Simple Method for Generating Gamma Variables"
    ## by George Marsaglia and Wai Wan Tsang.  
    ## ACM Transactions on Mathematical Software
    ## Vol 26, No 3, September 2000, pages 363-372.

    if shape >= 1.0
        d = shape - 1.0/3.0
        c = 1.0/sqrt(9.0*d)
        while true
            x = rand_normal(0, 1)
            v = 1.0 + c*x
            while v <= 0.0
                x = rand_normal(0, 1)
                v = 1.0 + c*x
            end
            v = v*v*v
            u = rand()
            xsq = x*x
            if u < 1.0 -.0331*xsq*xsq || log(u) < 0.5*xsq + d*(1.0 - v + log(v))
                return scale*d*v
            end
        end
    else
        g = rand_gamma(shape+1.0, 1.0)
        w = rand()
        return scale*g*pow(w, 1.0/shape)
    end
end

## return a random sample from a chi square distribution
## with the specified degrees of freedom
function rand_chi_square(dof)
    rand_gamma(0.5, 2.0*dof)
end

## return a random sample from an inverse gamma random variable
function rand_inverse_gamma(shape, scale)
    ## If X is gamma(shape, scale) then
    ## 1/Y is inverse gamma(shape, 1/scale)
    1.0 / rand_gamma(shape, 1.0 / scale)
end

## return a sample from a Weibull distribution
function rand_weibull(shape, scale)
    if shape <= 0.0
        error("Shape parameter must be positive")
    end
    if scale <= 0.0
        error("Scale parameter must be positive")
    end
    scale * pow(-log(rand()), 1.0 / shape)
end

## return a random sample from a Cauchy distribution
function rand_cauchy(median, scale)
    if scale <= 0.0
        error("Scale parameter must be positive")
    end
    p = rand()
    median + scale*tan(pi*(p - 0.5))
end

## return a random sample from a Student t distribution
function rand_student_t(dof)
    if dof <= 0
        error("Degrees of freedom must be positive")
    end

    ## See Seminumerical Algorithms by Knuth
    y1 = rand_normal(0, 1)
    y2 = rand_chi_square(dof)
    y1 / sqrt(y2 / dof)
end
 
## return a random sample from a Laplace distribution
## The Laplace distribution is also known as the double exponential distribution.
function rand_laplace(mean, scale)   
    if scale <= 0.0
        error("Scale parameter must be positive")
    end
    u = rand()
    if u < 0.5
        retval = mean + scale*log(2.0*u) 
    else
        retval = mean - scale*log(2*(1-u))
    end
    retval
end

## return a random sample from a log-normal distribution
function rand_log_normal(mu, sigma)
    return exp(rand_normal(mu, sigma))
end

## return a random sample from a beta distribution
function rand_beta(a, b)
    if a <= 0 || b <= 0
        error("Beta parameters must be positive")
    end
    
    ## There are more efficient methods for generating beta samples.
    ## However such methods are a little more efficient and much more complicated.
    ## For an explanation of why the following method works, see
    ## http://www.johndcook.com/distribution_chart.html#gamma_beta

    u = rand_gamma(a, 1.0)
    v = rand_gamma(b, 1.0)
    u / (u + v)
end
