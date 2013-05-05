#   Translation of minpack subroutine cstep
#   Dianne O'Leary   July 1991
#
#     Subroutine cstep
#
#     The purpose of cstep is to compute a safeguarded step for
#     a linesearch and to update an interval of uncertainty for
#     a minimizer of the function.
#
#     The parameter stx contains the step with the least function
#     value. The parameter stp contains the current step. It is
#     assumed that the derivative at stx is negative in the
#     direction of the step. If bracketed is set true then a
#     minimizer has been bracketed in an interval of uncertainty
#     with endpoints stx and sty.
#
#     The subroutine statement is
#
#       subroutine cstep(stx, fx, dgx,
#                        sty, fy, dgy,
#                        stp, f, dp,
#                        bracketed, stpmin, stpmax, info)
#
#     where
#
#       stx, fx, and dgx are variables which specify the step,
#         the function, and the derivative at the best step obtained
#         so far. The derivative must be negative in the direction
#         of the step, that is, dgx and stp-stx must have opposite
#         signs. On output these parameters are updated appropriately.
#
#       sty, fy, and dgy are variables which specify the step,
#         the function, and the derivative at the other endpoint of
#         the interval of uncertainty. On output these parameters are
#         updated appropriately.
#
#       stp, f, and dp are variables which specify the step,
#         the function, and the derivative at the current step.
#         If bracketed is set true then on input stp must be
#         between stx and sty. On output stp is set to the new step.
#
#       bracketed is a logical variable which specifies if a minimizer
#         has been bracketed. If the minimizer has not been bracketed
#         then on input bracketed must be set false. If the minimizer
#         is bracketed then on output bracketed is set true.
#
#       stpmin and stpmax are input variables which specify lower
#         and upper bounds for the step.
#
#       info is an integer output variable set as follows:
#         If info = 1,2,3,4,5, then the step has been computed
#         according to one of the five cases below. Otherwise
#         info = 0, and this indicates improper input parameters.
#
#     Argonne National Laboratory. MINPACK Project. June 1983
#     Jorge J. More', David J. Thuente

function cstep(stx::Real, fx::Real, dgx::Real,
               sty::Real, fy::Real, dgy::Real,
               stp::Real, f::Real, dg::Real,
               bracketed::Bool, stpmin::Real, stpmax::Real)

   info = 0

   #
   #     Check the input parameters for errors
   #

   if (bracketed && (stp <= min(stx, sty) || stp >= max(stx, sty))) ||
        dgx * (stp - stx) >= 0.0 || stpmax < stpmin
      throw(ArgumentError("Minimizer not bracketed"))
   end

   #
   #     Determine if the derivatives have opposite sign
   #

   sgnd = dg * (dgx / abs(dgx))

   #
   #     First case. A higher function value.
   #     The minimum is bracketed. If the cubic step is closer
   #     to stx than the quadratic step, the cubic step is taken,
   #     else the average of the cubic and quadratic steps is taken.
   #

   if f > fx
      info = 1
      bound = true
      theta = 3 * (fx - f) / (stp - stx) + dgx + dg
      s = max(theta, dgx, dg)
      gamma = s * sqrt((theta / s)^2 - (dgx / s) * (dg / s))
      if stp < stx
          gamma = -gamma
      end
      p = gamma - dgx + theta
      q = gamma - dgx + gamma + dg
      r = p / q
      stpc = stx + r * (stp - stx)
      stpq = stx + ((dgx / ((fx - f) / (stp - stx) + dgx)) / 2) * (stp - stx)
      if abs(stpc - stx) < abs(stpq - stx)
         stpf = stpc
      else
         stpf = stpc + (stpq - stpc) / 2
      end
      bracketed = true

      #
      # Second case. A lower function value and derivatives of
      # opposite sign. The minimum is bracketed. If the cubic
      # step is closer to stx than the quadratic (secant) step,
      # the cubic step is taken, else the quadratic step is taken.
      #

   elseif sgnd < 0.0
      info = 2
      bound = false
      theta = 3 * (fx - f) / (stp - stx) + dgx + dg
      s = max(theta, dgx, dg)
      gamma = s * sqrt((theta / s)^2 - (dgx / s) * (dg / s))
      if stp > stx
         gamma = -gamma
      end
      p = gamma - dg + theta
      q = gamma - dg + gamma + dgx
      r = p / q
      stpc = stp + r * (stx - stp)
      stpq = stp + (dg / (dg - dgx)) * (stx - stp)
      if abs(stpc - stp) > abs(stpq - stp)
         stpf = stpc
      else
         stpf = stpq
      end
      bracketed = true

      #
      # Third case. A lower function value, derivatives of the
      # same sign, and the magnitude of the derivative decreases.
      # The cubic step is only used if the cubic tends to infinity
      # in the direction of the step or if the minimum of the cubic
      # is beyond stp. Otherwise the cubic step is defined to be
      # either stpmin or stpmax. The quadratic (secant) step is also
      # computed and if the minimum is bracketed then the the step
      # closest to stx is taken, else the step farthest away is taken.
      #

   elseif abs(dg) < abs(dgx)
      info = 3
      bound = true
      theta = 3 * (fx - f) / (stp - stx) + dgx + dg
      s = max(theta, dgx, dg)
      #
      # The case gamma = 0 only arises if the cubic does not tend
      # to infinity in the direction of the step.
      #
      gamma = s * sqrt(max(0.0, (theta / s)^2 - (dgx / s) * (dg / s)))
      if stp > stx
          gamma = -gamma
      end
      p = gamma - dg + theta
      q = gamma + dgx - dg + gamma
      r = p / q
      if r < 0.0 && gamma != 0.0
         stpc = stp + r * (stx - stp)
      elseif stp > stx
         stpc = stpmax
      else
         stpc = stpmin
      end
      stpq = stp + (dg / (dg - dgx)) * (stx - stp)
      if bracketed
         if abs(stp - stpc) < abs(stp - stpq)
            stpf = stpc
         else
            stpf = stpq
         end
      else
         if abs(stp - stpc) > abs(stp - stpq)
            stpf = stpc
         else
            stpf = stpq
         end
      end

      #
      # Fourth case. A lower function value, derivatives of the
      # same sign, and the magnitude of the derivative does
      # not decrease. If the minimum is not bracketed, the step
      # is either stpmin or stpmax, else the cubic step is taken.
      #

   else
      info = 4
      bound = false
      if bracketed
         theta = 3 * (f - fy) / (sty - stp) + dgy + dg
         s = max(theta, dgy, dg)
         gamma = s * sqrt((theta / s)^2 - (dgy / s) * (dg / s))
         if stp > sty
             gamma = -gamma
         end
         p = gamma - dg + theta
         q = gamma - dg + gamma + dgy
         r = p / q
         stpc = stp + r * (sty - stp)
         stpf = stpc
      elseif stp > stx
         stpf = stpmax
      else
         stpf = stpmin
      end
   end

   #
   # Update the interval of uncertainty. This update does not
   # depend on the new step or the case analysis above.
   #

   if f > fx
      sty = stp
      fy = f
      dgy = dg
   else
      if sgnd < 0.0
         sty = stx
         fy = fx
         dgy = dgx
      end
      stx = stp
      fx = f
      dgx = dg
   end

   #
   # Compute the new step and safeguard it.
   #

   stpf = min(stpmax, stpf)
   stpf = max(stpmin, stpf)
   stp = stpf
   if bracketed && bound
      if sty > stx
         stp = min(stx + 0.66 * (sty - stx), stp)
      else
         stp = max(stx + 0.66 * (sty - stx), stp)
      end
   end

   return stx, fx, dgx, sty, fy, dgy, stp, f, dg, bracketed, info
end
