/*
Copyright 2024 Enzo Busseti

This file is part of Project Euromir.

Project Euromir is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

Project Euromir is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
Project Euromir. If not, see <https://www.gnu.org/licenses/>.

Minimal changes by hand to the f2c translation of the original FORTRAN code
from Lbfgsb3.0. See copyright notice therein.
*/


/* dcstep.f -- translated by f2c (version 20200916).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include <math.h>
#include <stdbool.h>

#include "fortran.h"

/* Subroutine */ void dcstep(
	double *stx,
	double *fx,
	double *dx,
	double *sty,
	double *fy,
	double *dy,
	double *stp,
	double *fp,
	double *dp,
	bool *brackt,
	double *stpmin,
	double *stpmax)
{
    /* System generated locals */
    double d__1, d__2, d__3;


    /* Local variables */
    double p, q, r__, s, sgnd, stpc, stpf, stpq, gamma, theta;

/*     ********** */

/*     Subroutine dcstep */

/*     This subroutine computes a safeguarded step for a search */
/*     procedure and updates an interval that contains a step that */
/*     satisfies a sufficient decrease and a curvature condition. */

/*     The parameter stx contains the step with the least function */
/*     value. If brackt is set to .true. then a minimizer has */
/*     been bracketed in an interval with endpoints stx and sty. */
/*     The parameter stp contains the current step. */
/*     The subroutine assumes that if brackt is set to .true. then */

/*           min(stx,sty) < stp < max(stx,sty), */

/*     and that the derivative at stx is negative in the direction */
/*     of the step. */

/*     The subroutine statement is */

/*       subroutine dcstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt, */
/*                         stpmin,stpmax) */

/*     where */

/*       stx is a double precision variable. */
/*         On entry stx is the best step obtained so far and is an */
/*            endpoint of the interval that contains the minimizer. */
/*         On exit stx is the updated best step. */

/*       fx is a double precision variable. */
/*         On entry fx is the function at stx. */
/*         On exit fx is the function at stx. */

/*       dx is a double precision variable. */
/*         On entry dx is the derivative of the function at */
/*            stx. The derivative must be negative in the direction of */
/*            the step, that is, dx and stp - stx must have opposite */
/*            signs. */
/*         On exit dx is the derivative of the function at stx. */

/*       sty is a double precision variable. */
/*         On entry sty is the second endpoint of the interval that */
/*            contains the minimizer. */
/*         On exit sty is the updated endpoint of the interval that */
/*            contains the minimizer. */

/*       fy is a double precision variable. */
/*         On entry fy is the function at sty. */
/*         On exit fy is the function at sty. */

/*       dy is a double precision variable. */
/*         On entry dy is the derivative of the function at sty. */
/*         On exit dy is the derivative of the function at the exit sty. */

/*       stp is a double precision variable. */
/*         On entry stp is the current step. If brackt is set to .true. */
/*            then on input stp must be between stx and sty. */
/*         On exit stp is a new trial step. */

/*       fp is a double precision variable. */
/*         On entry fp is the function at stp */
/*         On exit fp is unchanged. */

/*       dp is a double precision variable. */
/*         On entry dp is the the derivative of the function at stp. */
/*         On exit dp is unchanged. */

/*       brackt is an logical variable. */
/*         On entry brackt specifies if a minimizer has been bracketed. */
/*            Initially brackt must be set to .false. */
/*         On exit brackt specifies if a minimizer has been bracketed. */
/*            When a minimizer is bracketed brackt is set to .true. */

/*       stpmin is a double precision variable. */
/*         On entry stpmin is a lower bound for the step. */
/*         On exit stpmin is unchanged. */

/*       stpmax is a double precision variable. */
/*         On entry stpmax is an upper bound for the step. */
/*         On exit stpmax is unchanged. */

/*     MINPACK-1 Project. June 1983 */
/*     Argonne National Laboratory. */
/*     Jorge J. More' and David J. Thuente. */

/*     MINPACK-2 Project. November 1993. */
/*     Argonne National Laboratory and University of Minnesota. */
/*     Brett M. Averick and Jorge J. More'. */

/*     ********** */
    sgnd = *dp * (*dx / abs(*dx));
/*     First case: A higher function value. The minimum is bracketed. */
/*     If the cubic step is closer to stx than the quadratic step, the */
/*     cubic step is taken, otherwise the average of the cubic and */
/*     quadratic steps is taken. */
    if (*fp > *fx) {
	theta = (*fx - *fp) * 3. / (*stp - *stx) + *dx + *dp;
/* Computing MAX */
	d__1 = abs(theta), d__2 = abs(*dx), d__1 = max(d__1,d__2), d__2 = abs(
		*dp);
	s = max(d__1,d__2);
/* Computing 2nd power */
	d__1 = theta / s;
	gamma = s * sqrt(d__1 * d__1 - *dx / s * (*dp / s));
	if (*stp < *stx) {
	    gamma = -gamma;
	}
	p = gamma - *dx + theta;
	q = gamma - *dx + gamma + *dp;
	r__ = p / q;
	stpc = *stx + r__ * (*stp - *stx);
	stpq = *stx + *dx / ((*fx - *fp) / (*stp - *stx) + *dx) / 2. * (*stp 
		- *stx);
	if ((d__1 = stpc - *stx, abs(d__1)) < (d__2 = stpq - *stx, abs(d__2)))
		 {
	    stpf = stpc;
	} else {
	    stpf = stpc + (stpq - stpc) / 2.;
	}
	*brackt = true;
/*     Second case: A lower function value and derivatives of opposite */
/*     sign. The minimum is bracketed. If the cubic step is farther from */
/*     stp than the secant step, the cubic step is taken, otherwise the */
/*     secant step is taken. */
    } else if (sgnd < 0.) {
	theta = (*fx - *fp) * 3. / (*stp - *stx) + *dx + *dp;
/* Computing MAX */
	d__1 = abs(theta), d__2 = abs(*dx), d__1 = max(d__1,d__2), d__2 = abs(
		*dp);
	s = max(d__1,d__2);
/* Computing 2nd power */
	d__1 = theta / s;
	gamma = s * sqrt(d__1 * d__1 - *dx / s * (*dp / s));
	if (*stp > *stx) {
	    gamma = -gamma;
	}
	p = gamma - *dp + theta;
	q = gamma - *dp + gamma + *dx;
	r__ = p / q;
	stpc = *stp + r__ * (*stx - *stp);
	stpq = *stp + *dp / (*dp - *dx) * (*stx - *stp);
	if ((d__1 = stpc - *stp, abs(d__1)) > (d__2 = stpq - *stp, abs(d__2)))
		 {
	    stpf = stpc;
	} else {
	    stpf = stpq;
	}
	*brackt = true;
/*     Third case: A lower function value, derivatives of the same sign, */
/*     and the magnitude of the derivative decreases. */
    } else if (abs(*dp) < abs(*dx)) {
/*        The cubic step is computed only if the cubic tends to infinity */
/*        in the direction of the step or if the minimum of the cubic */
/*        is beyond stp. Otherwise the cubic step is defined to be the */
/*        secant step. */
	theta = (*fx - *fp) * 3. / (*stp - *stx) + *dx + *dp;
/* Computing MAX */
	d__1 = abs(theta), d__2 = abs(*dx), d__1 = max(d__1,d__2), d__2 = abs(
		*dp);
	s = max(d__1,d__2);
/*        The case gamma = 0 only arises if the cubic does not tend */
/*        to infinity in the direction of the step. */
/* Computing MAX */
/* Computing 2nd power */
	d__3 = theta / s;
	d__1 = 0., d__2 = d__3 * d__3 - *dx / s * (*dp / s);
	gamma = s * sqrt((max(d__1,d__2)));
	if (*stp > *stx) {
	    gamma = -gamma;
	}
	p = gamma - *dp + theta;
	q = gamma + (*dx - *dp) + gamma;
	r__ = p / q;
	if (r__ < 0. && gamma != 0.) {
	    stpc = *stp + r__ * (*stx - *stp);
	} else if (*stp > *stx) {
	    stpc = *stpmax;
	} else {
	    stpc = *stpmin;
	}
	stpq = *stp + *dp / (*dp - *dx) * (*stx - *stp);
	if (*brackt) {
/*           A minimizer has been bracketed. If the cubic step is */
/*           closer to stp than the secant step, the cubic step is */
/*           taken, otherwise the secant step is taken. */
	    if ((d__1 = stpc - *stp, abs(d__1)) < (d__2 = stpq - *stp, abs(
		    d__2))) {
		stpf = stpc;
	    } else {
		stpf = stpq;
	    }
	    if (*stp > *stx) {
/* Computing MIN */
		d__1 = *stp + (*sty - *stp) * .66;
		stpf = min(d__1,stpf);
	    } else {
/* Computing MAX */
		d__1 = *stp + (*sty - *stp) * .66;
		stpf = max(d__1,stpf);
	    }
	} else {
/*           A minimizer has not been bracketed. If the cubic step is */
/*           farther from stp than the secant step, the cubic step is */
/*           taken, otherwise the secant step is taken. */
	    if ((d__1 = stpc - *stp, abs(d__1)) > (d__2 = stpq - *stp, abs(
		    d__2))) {
		stpf = stpc;
	    } else {
		stpf = stpq;
	    }
	    stpf = min(*stpmax,stpf);
	    stpf = max(*stpmin,stpf);
	}
/*     Fourth case: A lower function value, derivatives of the same sign, */
/*     and the magnitude of the derivative does not decrease. If the */
/*     minimum is not bracketed, the step is either stpmin or stpmax, */
/*     otherwise the cubic step is taken. */
    } else {
	if (*brackt) {
	    theta = (*fp - *fy) * 3. / (*sty - *stp) + *dy + *dp;
/* Computing MAX */
	    d__1 = abs(theta), d__2 = abs(*dy), d__1 = max(d__1,d__2), d__2 = 
		    abs(*dp);
	    s = max(d__1,d__2);
/* Computing 2nd power */
	    d__1 = theta / s;
	    gamma = s * sqrt(d__1 * d__1 - *dy / s * (*dp / s));
	    if (*stp > *sty) {
		gamma = -gamma;
	    }
	    p = gamma - *dp + theta;
	    q = gamma - *dp + gamma + *dy;
	    r__ = p / q;
	    stpc = *stp + r__ * (*sty - *stp);
	    stpf = stpc;
	} else if (*stp > *stx) {
	    stpf = *stpmax;
	} else {
	    stpf = *stpmin;
	}
    }
/*     Update the interval which contains a minimizer. */
    if (*fp > *fx) {
	*sty = *stp;
	*fy = *fp;
	*dy = *dp;
    } else {
	if (sgnd < 0.) {
	    *sty = *stx;
	    *fy = *fx;
	    *dy = *dx;
	}
	*stx = *stp;
	*fx = *fp;
	*dx = *dp;
    }
/*     Compute the new step. */
    *stp = stpf;
} /* dcstep_ */

