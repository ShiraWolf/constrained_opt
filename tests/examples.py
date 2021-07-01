import numpy as np


def quadratic_func(Q, x, calc_hessian):
    # generic quadric function xT * Q * x
    val = np.dot(x, np.dot(Q, x))
    gradient = np.dot(Q + Q.transpose(), x)
    if calc_hessian:
        hessian = 2 * Q
        return val, gradient, hessian
    else:
        return val, gradient


def quadratic_func1(x, calc_hessian=False):
    Q = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    return quadratic_func(Q, x, calc_hessian)


def quadratic_func2(x, calc_hessian=False):
    Q = np.array([[5.0, 0.0],
                  [0.0, 1.0]])
    return quadratic_func(Q, x, calc_hessian)


def quadratic_func3(x, calc_hessian=False):
    Q1 = np.array([[np.sqrt(3) / 2, -0.5],
                   [0.5, np.sqrt(3) / 2]])
    Q2 = np.array([[5.0, 0.0],
                   [0.0, 1.0]])
    Q = np.dot(np.transpose(Q1), np.dot(Q2, Q1))
    return quadratic_func(Q, x, calc_hessian)


def rosenbrock_func(x, calc_hessian=False):
    assert len(x) == 2
    val = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    gradient = np.array([400 * (x[0] ** 3) - 400 * x[0] * x[1] + 2 * x[0] - 2, -200 * (x[0] ** 2) + 200 * x[1]])
    if calc_hessian:
        hessian = np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])
        return val, gradient, hessian
    else:
        return val, gradient


def lin_func(x, calc_hessian=False):
    a = np.array([2.0, 3.0])
    assert len(x) == len(a)
    val = np.dot(a, x)
    gradient = a
    if calc_hessian:
        hessian = np.zeros((len(a), len(a)))
        return val, gradient, hessian
    else:
        return val, gradient


def func_qp(x, calc_hessian=False):
    val = np.array(x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2)
    gradient = np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])
    if calc_hessian:
        hessian = 2 * np.eye(3)
        return val, gradient, hessian
    else:
        return val, gradient


def func_lp(x, calc_hessian=False):
    val = - np.array(x[0] + x[1])  # max(x + y) is -min(x+y)
    gradient = - np.array([1.0, 1.0])
    if calc_hessian:
        hessian = np.zeros((2, 2))
        return val, gradient, hessian
    else:
        return val, gradient


class LinInequalityConstraint:
    # implements a function-like constraint of the form ax <= b
    def __init__(self, a, b):
        self.a = a
        self.b = b
        # self.label = r'$'
        # if len(a) == 2:
        #     if a[0] != 0:
        #         self.label += str(a[0]) + 'x'
        #     if a[0] != 0:
        #         self.label += str(a[1]) + 'y'
        # elif len(a) == 3:

    # implements the function, not calculating hessian because it is zero in this case
    def __call__(self, x):
        assert len(x) == len(self.a)
        val = np.dot(self.a, x) - self.b
        grad = self.a
        return val, grad



class ConstrainedProblem:
    # just a struct of constrained problem parameters
    # TODO make this a parent class for LinInequalityConstraint
    def __init__(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs


def constrained_problem_qp():
    f0 = func_qp
    ineqConstraint1 = LinInequalityConstraint([-1, 0, 0], 0)  # -x <= 0
    ineqConstraint2 = LinInequalityConstraint([0, -1, 0], 0)  # -y <= 0
    ineqConstraint3 = LinInequalityConstraint([0, 0, -1], 0)  # -z <= 0
    ineq_constraints = [ineqConstraint1, ineqConstraint2, ineqConstraint3]
    eq_constraints_mat = np.array([[1, 1, 1]]) # x + y + z = 1
    eq_constraints_rhs = np.array([1])
    return ConstrainedProblem(f0, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)


def constrained_problem_lp():
    f0 = func_lp
    ineqConstraint1 = LinInequalityConstraint([-1, -1], -1)   # -y -x + 1 <= 0
    ineqConstraint2 = LinInequalityConstraint([0, 1], 1)    # y - 1 <= 0
    ineqConstraint3 = LinInequalityConstraint([1, 0], 2)    # x - 2 <= 0
    ineqConstraint4 = LinInequalityConstraint([0, -1], 0)    # -y <= 0
    ineq_constraints = [ineqConstraint1, ineqConstraint2, ineqConstraint3, ineqConstraint4]
    eq_constraints_mat = np.array([])
    eq_constraints_rhs = np.array([])
    return ConstrainedProblem(f0, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)

# def plot_surface_qp():
#     return 0
