import time
from copy import deepcopy

import numpy as np

from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer


def _li_fukushima(manifold, newx, s, diff):
    _eps = 1e-6
    sPs = manifold.inner_product(newx, s, s)
    sPdiff = manifold.inner_product(newx, s, diff)

    _mu = 0.
    if sPdiff < _eps * sPs:
        _mu = max(0., -sPdiff / sPs) + _eps
    return diff + _mu * s


def _powells_damping(manifold, newx, s, diff):
    _eps = 0.1
    sPs = manifold.inner_product(newx, s, s)
    sPdiff = manifold.inner_product(newx, s, diff)

    _mu = 1.
    if sPdiff < _eps * sPs:
        _denominator = sPs - sPdiff
        _mu = (
            (1 - _eps)
            * sPs
            / _denominator
        )
    return _mu * diff + (1 - _mu) * s


def _formula_bfgs(sPs, diffPdiff, sPdiff):
    return 1.


def _formula_dfp(sPs, diffPdiff, sPdiff):
    return 0.


def _formula_preconvex(sPs, diffPdiff, sPdiff):
    _mu = (
        sPs
        * diffPdiff
        / (sPdiff ** 2)
    )
    _theta = max(1. / (1. - _mu), -1e5)
    return (0.1 * _theta - 1) / (0.1 * _theta * (1 - _mu) - 1)


FORMULAS = {
    "BFGS": _formula_bfgs,
    "DFP": _formula_dfp,
    "preconvex": _formula_preconvex
}


REGULARIZATION_METHODS = {
    "LiFukushima": _li_fukushima,
    "Powell": _powells_damping,
}


class MemorylessQuasiNewton(Optimizer):
    def __init__(
        self,
        orth_value=np.inf,
        line_searcher=None,
        formula='BFGS',
        regularization_method='LiFukushima',
        xi: float=1.,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._orth_value = orth_value

        if line_searcher is None:
            self._line_searcher = LineSearchWolfe()
        else:
            self._line_searcher = line_searcher
        self.line_searcher = None
        self._calc_phi = FORMULAS[formula]
        self._regularization = REGULARIZATION_METHODS[regularization_method]
        self.xi = xi

    def run(self, problem, *, initial_point=None, reuse_line_searcher=False) -> OptimizerResult:
        manifold = problem.manifold
        objective = problem.cost
        gradient = problem.riemannian_gradient

        if not reuse_line_searcher or self.line_searcher is None:
            self.line_searcher = deepcopy(self._line_searcher)
        line_searcher = self.line_searcher

        if initial_point is None:
            x = manifold.random_point()
        else:
            x = initial_point

        if self._verbosity >= 1:
            print("Optimizing...")
        if self._verbosity >= 2:
            iteration_format_length = int(np.log10(self._max_iterations)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iteration_format_length}d"),
                    ("Cost", "+.16e"),
                    ("Gradient norm", ".8e"),
                ]
            )
        else:
            column_printer = printer.VoidPrinter()

        column_printer.print_header()

        cost = objective(x)
        grad = gradient(x)
        gradient_norm = manifold.norm(x, grad)
        Pgrad = problem.preconditioner(x, grad)
        gradPgrad = manifold.inner_product(x, grad, Pgrad)

        descent_direction = -Pgrad

        self._initialize_log(
            optimizer_parameters={
                "orth_value": self._orth_value,
                "line_searcher": line_searcher,
            },
        )

        iteration = 0
        step_size = np.nan
        start_time = time.time()

        while True:
            iteration += 1

            column_printer.print_row([iteration, cost, gradient_norm])
            self._add_log_entry(
                iteration=iteration,
                point=x,
                cost=cost,
                gradient_norm=gradient_norm,
            )
            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                gradient_norm=gradient_norm,
                iteration=iteration,
                step_size=step_size,
            )
            if stopping_criterion:
                if self._verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break

            df0 = manifold.inner_product(x, grad, descent_direction)
            if df0 >= 0:
                if self._verbosity >= 3:
                    print(
                        "Conjugate gradient info: got an ascent direction "
                        f"(df0 = {df0:.2f}), reset to the (preconditioned) "
                        "steepest descent direction."
                    )
                descent_direction = -Pgrad
                df0 = -gradPgrad

            step_size, newx = line_searcher.search(objective, manifold, x, descent_direction, cost, df0, gradient)

            newcost = objective(newx)
            newgrad = gradient(newx)
            newgradient_norm = manifold.norm(newx, newgrad)
            Pnewgrad = problem.preconditioner(newx, newgrad)
            newgradPnewgrad = manifold.inner_product(newx, newgrad, Pnewgrad)

            # Powell's restart strategy.
            oldgrad = manifold.transport(x, newx, grad)
            orth_grads = manifold.inner_product(newx, oldgrad, Pnewgrad) / newgradPnewgrad
            if abs(orth_grads) >= self._orth_value:
                descent_direction = -Pnewgrad
            else:
                descent_direction = manifold.transport(x, newx, descent_direction)
                s = manifold.transport(x, newx, step_size * descent_direction)
                diff = newgrad - oldgrad
                diff = self._regularization(manifold, newx, s, diff)

                sPs = manifold.inner_product(newx, s, s)
                diffPdiff = manifold.inner_product(newx, diff, diff)
                sPdiff = manifold.inner_product(newx, s, diff)

                gamma = max(1., sPdiff / diffPdiff)
                tau = min(1., diffPdiff / sPdiff)
                phi = self._calc_phi(sPs, diffPdiff, sPdiff)

                An = (
                    phi
                    * manifold.inner_product(newx, diff, Pnewgrad)
                    / sPdiff
                    - (1 / (gamma * tau) + phi * diffPdiff / sPdiff)
                    * manifold.inner_product(newx, s, Pnewgrad)
                    / sPdiff
                )

                Bn = (
                    phi
                    * manifold.inner_product(newx, s, Pnewgrad)
                    / sPdiff
                    + (1 - phi)
                    * manifold.inner_product(newx, diff, Pnewgrad)
                    / diffPdiff
                )

                eta = -gamma * Pnewgrad + gamma * An * s + gamma * self.xi * Bn * diff

                descent_direction = eta

            # Update the necessary variables for the next iteration.
            x = newx
            cost = newcost
            grad = newgrad
            Pgrad = Pnewgrad
            gradient_norm = newgradient_norm
            gradPgrad = newgradPnewgrad
        
        end_time = time.time()

        return self._return_result(
            start_time=start_time,
            point=x,
            cost=cost,
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=iteration,
            step_size=step_size,
            gradient_norm=gradient_norm,
        ), end_time - start_time


class LineSearchWolfe:
    def __init__(self, c1: float=1e-4, c2: float=0.999, strong=True):
        self.c1 = c1
        self.c2 = c2
        self.strong = strong

    def __str__(self):
        return 'Wolfe'

    def search(self, objective, man, x, d, f0, df0, gradient):
        strong = self.strong
        fc = [0]
        gc = [0]
        gval = [None]
        gval_alpha = [None]

        def phi(alpha):
            fc[0] += 1
            return objective(man.retraction(x, alpha * d))

        def derphi(alpha):
            newx = man.retraction(x, alpha * d)
            newd = man.transport(x, newx, d)
            gc[0] += 1
            gval[0] = gradient(newx)  # store for later use
            gval_alpha[0] = alpha
            return man.inner_product(newx, gval[0], newd)

        gfk = gradient(x)
        derphi0 = man.inner_product(x, gfk, d)

        stepsize = _scalar_search_wolfe(phi, derphi, self.c1, self.c2, maxiter=100, strong=strong)
        if stepsize is None:
            stepsize = 1e-6
        
        newx = man.retraction(x, stepsize * d)
        
        return stepsize, newx


def _scalar_search_wolfe(phi, derphi, c1=1e-4, c2=0.9, maxiter=100, strong=True):
    phi0 = phi(0.)
    derphi0 = derphi(0.)
    alpha0 = 0
    alpha1 = 1.0
    phi_a1 = phi(alpha1)
    phi_a0 = phi0
    derphi_a0 = derphi0
    for i in range(maxiter):
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or ((phi_a1 >= phi_a0) and (i > 1)):
            alpha_star, phi_star, derphi_star = _zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi, derphi, phi0, derphi0, c1, c2, strong=strong)
            break

        derphi_a1 = derphi(alpha1)
        if strong and (abs(derphi_a1) <= c2 * abs(derphi0)):
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = derphi_a1
            break
        if (not strong) and derphi_a1 >= c2 * derphi0:
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = derphi_a1
            break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = _zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi, derphi, phi0, derphi0, c1, c2, strong=strong)
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1
    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        print('The line search algorithm did not converge')
    
    return alpha_star


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi, phi0, derphi0, c1, c2, strong=True):
    """
    Part of the optimization algorithm in `_scalar_search_wolfe`.
    """
    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi
        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha
        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            # if abs(derphi_aj) <= c2 * abs(derphi0):
            if derphi_aj >= c2 * derphi0:
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin