#= samin.jl

  History: M. Creel's translation of Octave code samin.cc, also by Creel,
  which was originally based on Gauss code by E.G. Tsionas. A source
  for the Gauss code is http://web.stanford.edu/~doubleh/otherpapers/sa.txt

  The original Fortran code by W. Goffe is at http://www.degruyter.com/view/j/snde.1996.1.3/snde.1996.1.3.1020/snde.1996.1.3.1020.xml?format=INT

  Tsionas and Goffe agreed to MIT licensing of samin.jl in email
  messages to Creel.
  
  The original Fortran code accompanies  the article
  Goffe, William L. (1996) "SIMANN: A Global Optimization Algorithm
    using Simulated Annealing " Studies in Nonlinear Dynamics & Econometrics
    Oct96, Vol. 1 Issue 3.

  This Julia code uses the same names for control variables,
  for the most part. A notable difference is that the initial
  temperature can be found automatically to ensure that the active
  bounds when the temperature begins to reduce cover the entire
  parameter space (defined as a n-dimensional rectangle that is the
  Cartesian product of the(lb_i, ub_i), i = 1,2,..n. The code also
  allows for parameters to be restricted, by setting lb_i = ub_i,
  for the appropriate i.

  Other relevant articles (among many):
  Corana et. al., (1987) "Minimizing Multimodal Functions of Continuous
    Variables with the "Simulated Annealing" Algorithm",
    ACM Transactions on Mathematical Software, V. 13, N. 3.

  Goffe, et. al. (1994) "Global Optimization of Statistical Functions
    with Simulated Annealing", Journal of Econometrics,
    V. 60, N. 1/2.    

usage: x, obj, convergence, details = samin(f,
                                            x_init,
                                            lb,
                                            ub,
                                            nt,
                                            ns,
                                            rt,
                                            maxevals,
                                            neps,
                                            functol,
                                            paramtol,
                                            verbosity)
Arguments:
REQUIRED
* f: objective function
* x_init: starting value
* lb:  vector of lower bounds
* ub: vector of upper bounds
KEYWORDS
* nt:  integer: (default = 5) reduce temperature every nt*ns*dim(x_init) evaluations
* ns:  integer: (default = 5) adjust bounds every ns*dim(x_init) evaluations
* rt:  (0 < rt <1): (default = 0.5) geometric temperature reduction factor: when temp changes, new temp is t=rt*t 
* maxevals:  integer: limit on function evaluations
* neps:  integer: (default = 5) number of previous best values the final result is compared to
* functol: (> 0): (default = 1e-8) the required tolerance level for function value
                   comparisons
* paramtol: (> 0): (default = 1e-5) the required tolerance level for parameters
* verbosity: scalar: 0, 1, 2 or 3 (default = 1).
    * 0 = no screen output
    * 1 = only final results to screen
    * 2 = summary every temperature change, without param values
    * 3 = summary every temperature change, with param values
* coverage_ok: (0 or 1) (default = 0) 0: increase temperature until parameter space is
        covered by the trial values. 1: start decreasing temperature immediately
Returns:
* x: the minimizer
* obj: the value of f() at x
* convergence:
    0 if no convergence within maxevals function evaluations
    1 if normal convergence to a point interior to the parameter space
    2 if convergence to point very near bounds of parameter space
      (suggest re-running with looser bounds)
* details: a px3 matrix. p is the number of times improvements were found.
           The columns record information at the time an improvement was found
           * first: cumulative number of function evaluations
           * second: temperature
           * third: function value

Example: see samin_test
=#
function samin(obj_fn, x, lb, ub; nt=5, ns=5, rt=0.5, maxevals=1e6, neps=5, functol=1e-8, paramtol=1e-5, verbosity=1, coverage_ok=0)
    n = size(x,1) # dimension of parameter
    #  Set initial values
    nacc = 0 # total accepted trials
    t = 2.0 # temperature - will initially rise or fall to cover parameter space. Then it will fall
    converge = 0 # convergence indicator 0 (failure), 1 (normal success), or 2 (convergence but near bounds)
    # most recent values, to compare to when checking convergend
    fstar = typemax(Float64)*ones(neps)
    # Initial obj_value
    xopt = copy(x)
    f = obj_fn(x)
    fopt = copy(f) # give it something to compare to
    func_evals = 0 # total function evaluations (limited by maxeval)
    details = [func_evals t fopt]
    bounds = ub - lb
    # check for out-of-bounds starting values
    for i = 1:n
        if(( x[i] > ub[i]) || (x[i] < lb[i]))
            @printf("samin: initial parameter %d out of bounds\n", i)
            converge = 0
            return xopt, fopt, converge, details
        end 
    end
    # main loop, first increase temperature until parameter space covered, then reduce until convergence
    while (converge==0)
        # statistics to report at each temp change, set back to zero
        nup = 0
        nrej = 0
        nnew = 0
        ndown = 0
        lnobds = 0

        # repeat nt times then adjust temperature
        for m = 1:nt
            # repeat ns times, then adjust bounds
            nacp = zeros(n)
            for j = 1:ns
                # generate new point by taking last and adding a random value
                # to each of elements, in turn
                for h = 1:n
                    # new Sept 2011, if bounds are same, skip the search for that vbl.
                    # Allows restrictions without complicated programming
                    if (lb[h] != ub[h])
                        xp = copy(x)
                        xp[h] += (2.0 * rand() - 1.0) * bounds[h]
                        if((xp[h] < lb[h]) || (xp[h] > ub[h]))
                            xp[h] = lb[h] + (ub[h] - lb[h]) * rand()
                            lnobds += 1
                        end
                        # Evaluate function at new point
                        fp = obj_fn(xp)
                        func_evals += 1
                        #  Accept the new point if the function value decreases
                        if (fp <= f)
                            x = copy(xp)
                            f = copy(fp)
                            nacc += 1 # total number of acceptances
                            nacp[h] += 1 # acceptances for this parameter
                            nup += 1
                            #  If lower than any other point, record as new optimum
                            if(fp < fopt)
                                xopt = copy(xp)
                                fopt = copy(fp)
                                nnew +=1
                                details = [details; [func_evals t fp]]
                            end
                        # If the point is higher, use the Metropolis criteria to decide on
                        # acceptance or rejection.
                        else
                            p = exp(-(fp - f) / t)
                            if(rand() < p)
                                x = copy(xp)
                                f = copy(fp)
                                nacc += 1
                                nacp[h] += 1
                                ndown += 1
                            else
                                nrej += 1
                            end
                        end
                        # If maxevals exceeded, terminate the algorithm
                        if(func_evals >= maxevals)
                            if(verbosity >= 1)
                                println("================================================")
                                println("SAMIN results")
                                println("NO CONVERGENCE: MAXEVALS exceeded")
                                @printf("\n     Obj. value:  %16.5f\n\n", fopt)
                                if(verbosity >=2)
                                    println("       parameter      search width")
                                    for i=1:n
                                        @printf("%16.5f  %16.5f \n", xopt[i], bounds[i])
                                    end
                                end
                                println("================================================")
                            end
                            converge = 0
                            return xopt, fopt, converge, details
                        end
                    end
                end
            end
            #  Adjust bounds so that approximately half of all evaluations are accepted
            test = 0
            for i = 1:n
                if (lb[i] != ub[i])
                    ratio = nacp[i] / ns
                    if(ratio > 0.6) bounds[i] = bounds[i] * (1.0 + 2.0 * (ratio - 0.6) / 0.4) end
                    if(ratio < .4) bounds[i] = bounds[i] / (1.0 + 2.0 * ((0.4 - ratio) / 0.4)) end
                    # keep within initial bounds
                    if(bounds[i] >= (ub[i] - lb[i]))
                        bounds[i] = ub[i] - lb[i]
                        test += 1
                    end
                else
                    test = test + 1 # make sure coverage check passes for the fixed parameters
                end    
            end
            nacp = nacp - nacp # set back to zero
            # check if we cover parameter space, if we have yet to do so
            if (coverage_ok != 1) coverage_ok = (test == n) end
        end

        # intermediate output, if desired
        if(verbosity >= 2)
            println("samin: intermediate results before next temperature change")
            println("temperature: ", t)
            println("current best function value: ", fopt)
            println("total evaluations so far: ", func_evals)
            println("total moves since last temperature reduction: ", nup + ndown + nrej)
            println("downhill: ", nup)
            println("accepted uphill: ", ndown)
            println("rejected uphill: ", nrej)
            println("out of bounds trials: ", lnobds)
            println("new minima this temperature: ", nnew)
            println()
            if (verbosity > 2)
                println("       parameter      search width")
                for i=1:n
                    @printf("%16.5f  %16.5f \n", xopt[i], bounds[i])
                end
            end
            println()
        end        
        # Check for convergence, if we have covered the parameter space
        if (coverage_ok)

            # last value close enough to last neps values?
            fstar[1] = f
            test = 0
            for i=1:neps
                test += (abs(f - fstar[i]) > functol)
            end
            test = (test > 0) # if different from zero, function conv. has failed

            # last value close enough to overall best?
            if (((fopt - f) <= functol) && (!test))
                # check for bound narrow enough for parameter convergence
                for i = 1:n
                    if (bounds[i] > paramtol)
                        converge = 0 # no conv. if bounds too wide
                        break
                    else
                        converge = 1
                    end    
                end
            end
            # check if optimal point is near boundary of parameter space, and change message if so
            if (converge == 1) && (lnobds > 0)
                converge = 2
            end
            # Like to see the final results?
            if (converge > 0)
                if (verbosity >= 1)
                    println("================================================")
                    println("SAMIN results")
                    if (converge == 1)
                        println("==> Normal convergence <==")
                    end
                    if (converge == 2)
                        println("==> WARNING <==: Last point satisfies convergence criteria,")
                        println("but is near boundary of parameter space.")
                        println(lnobds, " out of  ", (nup+ndown+nrej), " evaluations were out of bounds in the last round.")
                        println("Expand bounds and re-run, unless this is a constrained minimization.")
                    end
                    @printf("\n     Obj. value:  %16.5f\n\n", fopt)
                    println("       parameter      search width")
                    for i=1:n
                        @printf("%16.5f  %16.5f \n", xopt[i], bounds[i])
                    end    
                    println("================================================")
                end
            end
            # Reduce temperature, record current function value in the
            # list of last "neps" values, and loop again
            t = rt * t
            for i = neps:-1:2
                fstar[i] = fstar[i-1]
            end    
            f = copy(fopt)
            x = copy(xopt)
        else  # coverage not ok - increase temperature quickly to expand search area
            t = 10. * t
            for i = neps:-1:2
                fstar[i] = fstar[i-1]
            end    
            f = fopt
            x = xopt
        end
    end
    return xopt, fopt, converge, details
end
