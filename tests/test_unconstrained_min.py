import unittest
import numpy as np
from src import unconstrained_min
from src import utils
import examples


class TestGradDescend(unittest.TestCase):

    def test_quad_min(self):
        x0 = np.array([1.0, 1.0])
        max_iter = 500
        param_tol = 10**-8
        obj_tol = 10**-12
        init_step_len = 0.1
        slope_ratio = 1e-4
        back_track_factor = 0.2
        for dir_selection_method in ['nt', 'bfgs']:
            for func, func_str in [(examples.quadratic_func1, 'Quad Function 1'),
                                    (examples.quadratic_func2, 'Quad Function 2'),
                                    (examples.quadratic_func3, 'Quad Function 3')]:
                full_path, is_converged, report_str = unconstrained_min.line_search(func, x0, obj_tol, param_tol, max_iter, dir_selection_method, init_step_len, slope_ratio, back_track_factor)
                if is_converged == 1:
                    report_str = report_str + ", Convergence achieved!"
                else:
                    report_str = report_str + ", Convergence failed!"
                title_str = func_str + ', Direction Algorithm - ' + dir_selection_method
                utils.make_report(report_str, title_str, func, full_path)

    def test_rosenbrock_min(self):
        func = examples.rosenbrock_func
        x0 = np.array([2.0, 2.0])
        max_iter = 14000
        param_tol = 10**-8
        obj_tol = 10**-7
        init_step_len = 0.001
        slope_ratio = 1e-4
        back_track_factor = 0.2
        func_str = 'Rosenbrock Function'
        for dir_selection_method in ['nt', 'bfgs']:
            full_path, is_converged, report_str = unconstrained_min.line_search(func, x0, obj_tol, param_tol, max_iter, dir_selection_method, init_step_len,
                                                                                slope_ratio, back_track_factor)

            if is_converged == 1:
                report_str = report_str + ", Convergence achieved!"
            else:
                report_str = report_str + ", Convergence failed!"
            title_str = func_str + ', Direction Algorithm - ' + dir_selection_method
            utils.make_report(report_str, title_str, func, full_path)

    def test_lin_min(self):
        func = examples.lin_func
        x0 = np.array([1.0, 1.0])
        init_step_len = 0.1
        slope_ratio = 1e-4
        back_track_factor = 0.2
        max_iter = 100
        param_tol = 10**-8
        obj_tol = 10**-7
        for dir_selection_method in ['nt', 'bfgs']:
            full_path, is_converged, report_str = unconstrained_min.line_search(func, x0, obj_tol, param_tol, max_iter, dir_selection_method, init_step_len, slope_ratio, back_track_factor)
            title_str = 'Lin_Function_Report'
            if is_converged == 1:
                report_str = report_str + ", Convergence achieved!"
            else:
                report_str = report_str + ", Convergence failed!"
            utils.make_report(report_str, title_str, func, full_path)


if __name__ == '__main__':
    unittest.main()