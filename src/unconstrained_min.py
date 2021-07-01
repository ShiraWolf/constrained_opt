#

import numpy as np
from src import utils


def line_search(f, x0, obj_tol, param_tol, max_iter, dir_selection_method, init_step_len, slope_ratio, back_track_factor):
    full_path = np.zeros([max_iter+1, len(x0)])
    full_path[0, :] = x0
    # sample_delta = param_tol / 2  # delta over which the numeric derivative is calculated
    x_current = x_new = x0
    func_val_current, grad, hess = f(x_current, calc_hessian=True)
    Bk = np.eye(len(x0))  # initial BFGS matrix, in this way we only calculate hessian once
    i = 0
    is_converged = 0
    while i < max_iter and is_converged == 0:
        # determine direction
        if dir_selection_method == 'gd':
            pk = -grad  # previously calculated
            # pk = -numeric_gradient(f, x_prev, sample_delta)
        elif dir_selection_method == 'nt':
            pk = newton_dir(f, x_current)
        elif dir_selection_method == 'bfgs':
            pk = np.linalg.solve(Bk, -grad)
        else:
            raise Exception("direction selected is not supported")

        # apply wolfe condition on step size
        x_new = x_current + init_step_len * pk
        func_val_new, grad_new = f(x_new)
        step_size = init_step_len
        while func_val_new > func_val_current + slope_ratio * step_size * np.dot(grad, pk) or np.isnan(func_val_new):
            # backtrack
            step_size *= back_track_factor
            # update x and obj function
            x_new = x_current + step_size * pk
            func_val_new, grad_new = f(x_new)


        # update new point in our full_path buffer to plot convergence later
        full_path[i+1, :] = x_new
        # calculate step size taken
        step_taken = np.sqrt(np.dot(x_new-x_current, x_new-x_current))
        # calculate obj function change
        val_change = np.abs(func_val_new-func_val_current)
        # print report to console
        report_str = utils.report_iteration(i, x_new, func_val_new, step_taken, val_change)
        # check convergence
        if step_taken < param_tol or val_change < obj_tol:
            # if we reach this then convergence is successful
            is_converged = 1
            print("Convergence achieved at iteration: " + str(i) + ", Function value: "
                  + str(func_val_new) + ", Args: " + str(x_new))
            full_path = np.delete(full_path, (np.arange(i + 1, len(full_path))), 0)
            return full_path, is_converged, report_str

        # prepare next iteration
        if dir_selection_method == 'bfgs':
            # update Bk if we are in BFGS line search
            s = x_new - x_current
            y = grad_new - grad
            Bk = Bk - (Bk @ (np.outer(s, np.transpose(s)) @ Bk)) / (np.dot(s, Bk @ s)) + np.outer(y, np.transpose(y)) / np.dot(y, s)

        func_val_current = func_val_new
        grad = grad_new
        x_current = x_new
        i += 1
    # if we reach this then we exceeded max number of iterations
    print("Convergence Failed , reached function value: "
          + str(np.array([func_val_new])) + ", at location: " + str(x_new))
    return full_path, is_converged, report_str


def numeric_gradient(f, x, delta):
    # not used, wrote for fun and to verify gradient calculations
    f_prime = np.zeros([len(x)])
    for i in range(len(x)): # TODO vectorize with numpy for efficiency
        # create vector of arguments
        xd = np.zeros_like(x)
        xd[i] = delta
        # calculate partial derivative numerically
        f_plus, g = f(x + xd)
        f_minus, g = f(x - xd)
        f_prime[i] = (f_plus - f_minus) / (2 * delta)
    return f_prime


def newton_dir(f, x):
    # need to solve linear equations df/dxi = -sum_j(d^2f/(dx_i*dx_j) * p_j) to get p_j
    # to do this we need second derivative/hessian matrix
    f_val, grad, hessian = f(x, calc_hessian=True)
    p = np.linalg.solve(hessian, -grad)
    return p
