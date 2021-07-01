
import numpy as np
from src import utils
from src import unconstrained_min


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    # variable eq_constraints_rhs is left unused because we assume feasible start and remain feasible
    t = 1
    mu = 10
    ineq_tol = 1e-9  # tolerance for m/t
    m = len(ineq_constraints)
    func_to_minimize = LogBarrier(func, ineq_constraints, t)
    full_path = np.array([x0])
    max_iter = 4000
    newton_tol = ineq_tol/10  # tolerance for newton decrement
    n_outer_loop = 0
    while m/t >= ineq_tol:
        n_outer_loop += 1
        print("\n")
        print("Minimizing log-barrier function with t = " + str(t) + ":")
        print("\n")
        # find minimizer
        tmp_path, tmp_is_converged, tmp_report_str = newton_constrained_method(func_to_minimize, x0, newton_tol, max_iter,  eq_constraints_mat)
        if not tmp_is_converged:
            full_path = np.append(full_path, tmp_path[1:, :], axis=0)
            final_val, grad = func_to_minimize(full_path[-1, :])
            report_str = "Convergence Failed in interior loop, Reached X = " + str(tmp_path[-1, :]) + ", Objective function value:" + str(final_val/t) + '\n'
            report_str += 't value: ' + str(t)
            return full_path, report_str
        # update t
        t *= mu
        func_to_minimize.t = t
        # update x0 to the last convergence point
        x0 = tmp_path[-1, :]
        full_path = np.append(full_path, tmp_path[1:, :], axis=0)
    # report convergence
    final_val, grad = func_to_minimize(full_path[-1, :])
    report_str = "Converged at X = " + str(tmp_path[-1, :]) + ", Objective function value:" + str(final_val/t) + ", m/t sub-optimality:" + str(m/t) + '\n'
    report_str += 'Number of outer iterations: ' + str(n_outer_loop) + ', number of total inner iterations: ' + str(full_path.shape[0])
    report_str += ', final t value: ' + str(t)
    return full_path, report_str


def newton_constrained_method(f, x0, newton_tol, max_iter,  eq_constraints_mat):
    full_path = np.zeros([max_iter + 1, len(x0)])
    full_path[0, :] = x0
    # newton method, assuming equality constraints with feasible start
    is_converged = 0
    i = 0
    x = x0
    back_track_factor = 0.2
    wolfe_slope_ratio = 1e-4
    num_eq_constraints = np.shape(eq_constraints_mat)[0]
    # convergence loop
    while i < max_iter and is_converged == 0:
        func_val, grad, hess = f(x, calc_hessian=True)
        # repeatedly solve the local quadratic problem with the equality constraints of preserving feasibility
        # assuming feasible start so no accounting for residual
        if num_eq_constraints > 0:
            kkt_mat = np.append(hess, np.transpose(eq_constraints_mat), axis=1)
            tmp_mat = np.append(eq_constraints_mat, np.zeros((num_eq_constraints, num_eq_constraints)))
            kkt_mat = np.append(kkt_mat, [tmp_mat], axis=0)
            kkt_rhs = np.append(-grad, np.zeros(num_eq_constraints))
        else:
            kkt_mat = hess
            kkt_rhs = -grad
        sol = np.linalg.solve(kkt_mat, kkt_rhs)
        pk = sol[0:len(x0)]        # step direction
        # w = sol[len(x0):]          # new lagrange multipliers, not needed in this method

        # apply wolfe condition on step size
        step_size = 1  # initial un-damped step length
        x_new = x + step_size * pk
        func_val_new, grad_new = f(x_new)
        while func_val_new > func_val + wolfe_slope_ratio * step_size * np.dot(grad, pk) or np.isnan(func_val_new):
            # backtrack
            step_size *= back_track_factor
            # update x and obj function
            x_new = x + step_size * pk
            func_val_new, grad_new = f(x_new)
        # save x
        full_path[i + 1, :] = x_new
        # calculate newton decrement
        newton_decrement = np.sqrt(np.dot(pk, np.dot(hess, pk)))
        # calculate obj function change
        val_change = np.abs(func_val_new-func_val)
        # print report to console
        report_str = utils.report_iteration(i, x_new, func_val_new, newton_decrement, val_change)
        # check convergence
        if (newton_decrement**2)/2 < newton_tol:
            is_converged = 1
            full_path = np.delete(full_path, (np.arange(i + 1, len(full_path))), 0)
            print("Convergence achieved at iteration: " + str(i) + ", Function value: "
                  + str(func_val_new) + ", Args: " + str(x_new))
            return full_path, is_converged, report_str
        # prepare next iteration
        x = x_new
        i += 1
    # if we reached here convergence has failed

    return full_path, is_converged, report_str


class LogBarrier:
    # function-like class for calling a function with log barrier
    def __init__(self, func, ineq_constraints, t):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.t = t

    def __call__(self, x, calc_hessian=False):
        outputs = self.func(x, calc_hessian)
        val = outputs[0]*self.t
        grad = outputs[1]*self.t
        if calc_hessian:
            hessian = outputs[2]*self.t
        for f in self.ineq_constraints:
            val_i, grad_i = f(x)
            val -= np.log(-val_i)
            # from lecture 9 slide 7, with hessian of constraints = 0:
            grad -= grad_i / val_i
            if calc_hessian:
                hessian += np.outer(grad_i, np.transpose(grad_i)) / (val_i ** 2)  # + 0
        if calc_hessian:
            return val, grad, hessian
        else:
            return val, grad


# NOT FINISHED, NOT USED, WROTE FOR UNFEASIBLE START, WHICH IS NOT THE CASE IN THIS EXCERCISE
def primal_dual_method(f, x0, res_tol, max_iter, eq_constraints_mat, eq_constraints_rhs):
    # this method does not assume feasible start
    x = x0
    w_prev = []
    backtrack_factor = 0.2
    alpha = 0.25
    is_converged = 0
    i = 0
    num_constraints = len(eq_constraints_rhs)
    while i < max_iter and is_converged == 0:
        val, grad, hess = f(x, calc_hessian=True)
        # solve linear system in order get primal and dual directions
        kkt_mat = np.append(hess,np.transpose(eq_constraints_mat),axis=1)
        tmp_mat = np.append(eq_constraints_mat,np.zeros((num_constraints, num_constraints)))
        kkt_mat = np.append(kkt_mat, [tmp_mat], axis=0)
        kkt_rhs = np.append(-grad, eq_constraints_rhs)
        sol = np.linalg.solve(kkt_mat, kkt_rhs)
        pk = sol[0:len(x0)]        # primal solution
        w = sol[len(x0):]          # dual solution
        # line search along direction pk (wolfe condition) on residual

        if len(w_prev) == 0:
            # first iteration, no change in w
            w_prev = w
        delta_nu = w - w_prev  # change in lagrange multipliers
        t = 1
        residual = calc_residual(grad, eq_constraints_mat, eq_constraints_rhs, x, w)
        residual_t = calc_residual(grad, eq_constraints_mat, eq_constraints_rhs, x + t * pk, w + t * delta_nu)
        while (1 - alpha*t)*np.sqrt(np.dot(residual, residual)) < np.sqrt(np.dot(residual_t, residual_t)):
            t *= backtrack_factor
            residual_t = calc_residual(grad, eq_constraints_mat, eq_constraints_rhs, x + t * pk, w + t * delta_nu)

        # update x and w
        x += pk * t
        w += delta_nu * t
        # check convergence
        if np.dot(eq_constraints_mat, x) == eq_constraints_rhs and residual_t < res_tol:
            is_converged = 1

        # prepare for next iteration
        w_prev = w


def calc_residual(gradient, eq_constraints_mat, eq_constraints_rhs, x, v):
    # calculate residual
    r_dual = gradient + np.dot(np.transpose(eq_constraints_mat), v)
    r_primal = np.dot(eq_constraints_mat, x) - eq_constraints_rhs
    return np.append(r_dual, r_primal)
