import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from fpdf import FPDF  # fpdf class
import os
import datetime


class PDF(FPDF):
    # class for pdf file
    def titles(self, title_str, x, y):
        self.set_xy(x, y)
        self.set_font('Arial', 'B', 16)
        self.set_text_color(220, 50, 50)
        self.cell(w=210.0, h=40.0, align='C', txt=title_str, border=0)

    def texts(self, x, y, txt, size=10):
        self.set_xy(x, y)
        self.set_text_color(76.0, 32.0, 250.0)
        self.set_font('Arial', '', size)
        self.multi_cell(0, 10, txt)


def make_report(report_str, title_str, f, full_path, ineq_constraints=None, eq_constraints_mat=None, eq_constraints_rhs=None):
    # open figure
    fig = plt.figure(figsize=(6, 10))
    # determine dimentions, full_path dimentions are (n_iter , len(x))
    x_dim = full_path.shape[1]
    if x_dim == 2:
        ax1 = fig.add_subplot(211)
        # plot convergence path
        plot_2d_contours_and_path(f, full_path, ax1, ineq_constraints)
    elif x_dim == 3:
        ax1 = fig.add_subplot(211, projection='3d')
        # plot convergence path
        plot_3d_feasible_region_and_path(ineq_constraints, eq_constraints_mat, eq_constraints_rhs, ax1, full_path)
        # ax1.legend() # commented because of a library bug in matplotlib, can work if running with breakpoint
    else:
        raise Exception("cannot make plot, Unsupported dimension")

    # plot func value vs number of iterations
    ax2 = fig.add_subplot(212)
    plot_objective_vs_iterations(f, full_path, ax2)

    # save figure
    # path tp save:
    dtime = datetime.datetime.now()
    fig_fname = 'results' + os.path.sep + title_str + '.png'

    fig.savefig(fig_fname)
    # add figure to PDF
    pdf = PDF()
    pdf.add_page()
    pdf.image(fig_fname, x=10, y=20, w=170, h=220)
    pdf.titles(title_str, 0, 0)
    pdf.texts(10.0, 240.0, report_str, size=10)
    time_str = str(dtime.day) + '_' + str(dtime.month) + '_' + str(dtime.year) + '_' + str(dtime.hour) + '_' + str(dtime.minute) + '_' + str(dtime.second) + '_'
    pdf_fname = 'results' + os.path.sep + time_str + title_str + '.pdf'
    pdf.output(pdf_fname, 'F')
    # delete figure
    os.remove(fig_fname)


def plot_2d_contours_and_path(f0, full_path, ax, ineq_constraints):
    # determine x bounds
    x_lower = np.min(full_path[:, 0])
    x_upper = np.max(full_path[:, 0])
    x_range = x_upper - x_lower
    x_lower -= x_range * 0.3
    x_upper += x_range * 0.3
    # determine y bounds
    y_lower = np.min(full_path[:, 1])
    y_upper = np.max(full_path[:, 1])
    y_range = y_upper - y_lower
    y_lower -= y_range * 0.7
    y_upper += y_range * 0.7
    # make meshgrid
    delta = np.max([y_upper - y_lower, x_upper - x_lower]) / 300
    x = np.arange(x_lower, x_upper, delta)
    y = np.arange(y_lower, y_upper, delta)
    X, Y = np.meshgrid(x, y)
    xx = np.array([X, Y])
    Z = np.array(np.zeros(xx.shape[1:]))
    F = np.array(np.ones(xx.shape[1:]))  # feasible region initialization
    for ii in range(xx.shape[1]):
        for jj in range(xx.shape[2]):
            # unfortunately I have to have this inefficient loop here because I did not implement the function to be easily vectorized with numpy, TODO optimize
            Z[ii, jj], grad = f0(xx[:, ii, jj])
            if ineq_constraints is not None:
                # feasible region
                for f in ineq_constraints:
                    val, grad = f(xx[:, ii, jj])
                    F[ii, jj] *= (val <= 0)

    # plot contours
    CS1 = ax.contour(X, Y, Z)
    ax.clabel(CS1, inline=True, fontsize=10)
    arrow_width = np.max([y_upper - y_lower, x_upper - x_lower]) / 600
    draw_2d_path(full_path, ax, arrow_width)
    # plot feasible region
    ax.imshow(F, origin="lower", cmap="Greys", alpha=0.3, extent=(x_lower, x_upper, y_lower, y_upper))  # TODO plot lines
    ax.set_title('Convergence Path')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')


def draw_2d_path(full_path, ax, width):
    # annotate start and end points
    ax.annotate("Start: ({:.2f}, {:.2f})".format(full_path[0, 0], full_path[0, 1]), (full_path[0, 0], full_path[0, 1]))
    ax.annotate("End: ({:.2f}, {:.2f})".format(full_path[-1, 0], full_path[-1, 1]), (full_path[-1, 0], full_path[-1, 1]))
    # draw path, TODO rescale the colormap in relation to distance and not iteration index
    c_map = plt.cm.jet
    c_norm = colors.Normalize(0, vmax=len(full_path) - 1)
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=c_map)
    for kk in range(len(full_path) - 1):
        color_val = scalar_map.to_rgba(kk)
        x_start = full_path[kk, 0]
        y_start = full_path[kk, 1]
        dx = full_path[kk + 1, 0] - full_path[kk, 0]
        dy = full_path[kk + 1, 1] - full_path[kk, 1]
        ax.arrow(x_start, y_start, dx, dy, width=width, color=color_val, head_width=7 * width)


def draw_3d_path(full_path, ax):
    # annotate start and end points
    x_start, y_start, z_start = full_path[0, 0], full_path[0, 1], full_path[0, 2]
    x_end, y_end, z_end = full_path[-1, 0], full_path[-1, 1], full_path[-1, 2]
    text = "Start: ({:.2f}, {:.2f}, {:.2f})".format(x_start, y_start, z_start)
    ax.text(x_start, y_start, z_start, text, color='red')
    text = "End: ({:.2f}, {:.2f}, {:.2f})".format(x_end, y_end, z_end)
    ax.text(x_end, y_end, z_end, text, color='red')
    # draw path
    X, Y, Z = zip(*full_path[:-1, :])
    U, V, W = zip(*(full_path[1:, :] - full_path[:-1, :]))
    cest2 = ax.quiver(X, Y, Z, U, V, W, color='red')
    ax.set_title('Convergence Path')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    cest2.set_label('Path')


def report_iteration(iter_num, xi, func_val, step_len, obj_val_change):
    np.set_printoptions(precision=2)
    report_string = "Iteration number: " + str(iter_num) + " New location: " + str(xi) + ", New value: " + str(np.array([func_val])) \
                    + ", Step length: " + str(np.array([step_len])) + ", Objective function change: " + str(np.array([obj_val_change]))
    print(report_string)
    return report_string


def plot_objective_vs_iterations(f, full_path, ax):
    obj_vals = np.zeros(len(full_path))
    for i in range(len(full_path)):
        obj_vals[i], grad = f(full_path[i, :])
    ax.plot(np.arange(0, len(full_path)), obj_vals)
    ax.set_title('Obj Function vs Iteration Num')
    ax.set_xlabel('Num Iterations')
    ax.set_ylabel('Function Value')
    ax.grid(b=True, which='both', axis='both')


def plot_3d_feasible_region_and_path(ineq_constraints, eq_constraints_mat, eq_constraints_rhs, ax, full_path):
    # constraints are function of numpy array x (3d)
    # determine bounds

    if True:
        # temporary non-generic plot for our qp problem, TODO make generic version
        triangles = [((0, 0, 1), (0, 1, 0), (1, 0, 0))]
        cest1 = ax.add_collection(Poly3DCollection(triangles, alpha=0.4))
        cest1.set_label('Feasible Region')
    else:
        # generic way to plot feasible region, but less pretty so not using it for submission
        # determine bounds
        xyz_lower = np.min(full_path)
        xyx_upper = np.max(full_path)
        tmp_range = xyx_upper - xyz_lower
        xyz_lower -= tmp_range * 0.3
        xyx_upper += tmp_range * 0.3
        x_limits = y_limits = z_limits = [xyz_lower, xyx_upper]
        # inequality constraints
        for f in ineq_constraints:
            helper_plot_hyperplane(ax, f.a, f.b, x_limits, y_limits, z_limits, 'r')
        # equality constraints
        if eq_constraints_mat is not None:
            n_eq_constraints = eq_constraints_mat.shape[0]
            for i in range(n_eq_constraints):
                a = eq_constraints_mat[i, :]
                b = eq_constraints_rhs[i]
                helper_plot_hyperplane(ax, a, b, x_limits, y_limits, z_limits, 'b')
        # set limits
        ax.set_xlim(xyz_lower, xyx_upper)
        ax.set_ylim(xyz_lower, xyx_upper)
        ax.set_zlim(xyz_lower, xyx_upper)
    # draw convergence path
    draw_3d_path(full_path, ax)


def helper_plot_hyperplane(ax, a, b, x_range, y_range, z_range, color):
    # draw 3d hyper of the form ax = b
    # a and x are vectors of length 3
    delta = np.max([x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]]) / 500
    if a[0] != 0:
        yy = np.arange(y_range[0], y_range[1], delta)
        zz = np.arange(z_range[0], z_range[1], delta)
        Y, Z = np.meshgrid(yy, zz)
        X = (b - a[1] * Y - a[2] * Z) / a[0]
        ax.plot_surface(X, Y, Z, color=color)
    elif a[1] != 0:
        xx = np.arange(x_range[0], x_range[1], delta)
        zz = np.arange(z_range[0], z_range[1], delta)
        X, Z = np.meshgrid(xx, zz)
        Y = (b - a[0] * X - a[2] * Z) / a[1]
        ax.plot_surface(X, Y, Z, color=color)
    elif a[2] != 0:
        xx = np.arange(x_range[0], x_range[1], delta)
        yy = np.arange(y_range[0], y_range[1], delta)
        X, Y = np.meshgrid(xx, yy)
        Z = (b - a[0] * X - a[1] * Y) / a[2]
        ax.plot_surface(X, Y, Z, color=color)
