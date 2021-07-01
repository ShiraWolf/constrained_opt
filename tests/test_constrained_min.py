import unittest
import numpy as np
from src import constrained_min
from src import utils
import examples


class TestConstrainedMin(unittest.TestCase):

    def test_qp(self):
        title_str = 'Constrained Quadratic Programing Report'
        x0 = np.array([0.1, 0.2, 0.7])
        problemStruct = examples.constrained_problem_qp()
        full_path, report_str = constrained_min.interior_pt(problemStruct.func, problemStruct.ineq_constraints,
                                                            problemStruct.eq_constraints_mat, problemStruct.eq_constraints_rhs, x0)

        utils.make_report(report_str, title_str, problemStruct.func, full_path, problemStruct.ineq_constraints, problemStruct.eq_constraints_mat,
                          problemStruct.eq_constraints_rhs)
        return 0

    def test_lp(self):
        title_str = 'Constrained Linear Programing Report'
        x0 = np.array([0.5, 0.75])
        problemStruct = examples.constrained_problem_lp()
        full_path, report_str = constrained_min.interior_pt(problemStruct.func, problemStruct.ineq_constraints,
                                                            problemStruct.eq_constraints_mat, problemStruct.eq_constraints_rhs, x0)

        utils.make_report(report_str, title_str, problemStruct.func, full_path, problemStruct.ineq_constraints)
        return 0


if __name__ == '__main__':
    unittest.main()
