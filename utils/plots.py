import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skopt.acquisition import gaussian_lcb, gaussian_ei, gaussian_pi


def plot_rf(res, **kwargs):
    """Plots the optimization results and the gaussian process
    for 1-D objective functions.

    Parameters
    ----------
    res :  `OptimizeResult`
        The result for which to plot the gaussian process.

    ax : `Axes`, optional
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    n_calls : int, default: -1
        Can be used to evaluate the model at call `n_calls`.

    objective : func, default: None
        Defines the true objective function. Must have one input parameter.

    n_points : int, default: 1000
        Number of data points used to create the plots

    noise_level : float, default: 0
        Sets the estimated noise level

    show_legend : boolean, default: True
        When True, a legend is plotted.

    show_title : boolean, default: True
        When True, a title containing the found minimum value
        is shown

    show_acq_func : boolean, default: False
        When True, the acquisition function is plotted

    show_next_point : boolean, default: False
        When True, the next evaluated point is plotted

    show_observations : boolean, default: True
        When True, observations are plotted as dots.

    show_mu : boolean, default: True
        When True, the predicted model is shown.

    Returns
    -------
    ax : `Axes`
        The matplotlib axes.
    """
    ax = kwargs.get("ax", None)
    n_calls = kwargs.get("n_calls", -1)
    objective = kwargs.get("objective", None)
    noise_level = kwargs.get("noise_level", 0)
    show_legend = kwargs.get("show_legend", True)
    show_title = kwargs.get("show_title", True)
    show_acq_func = kwargs.get("show_acq_func", False)
    show_next_point = kwargs.get("show_next_point", False)
    show_observations = kwargs.get("show_observations", True)
    show_mu = kwargs.get("show_mu", True)
    n_points = kwargs.get("n_points", 1000)

    if ax is None:
        ax = plt.gca()
    n_dims = res.space.n_dims
    assert n_dims == 1, "Space dimension must be 1"
    dimension = res.space.dimensions[0]
    x, x_model = _evenly_sample(dimension, n_points)
    x = x.reshape(-1, 1)
    x_model = x_model.reshape(-1, 1)
    if res.specs is not None and "args" in res.specs:
        n_random = res.specs["args"].get('n_random_starts', None)
        acq_func = res.specs["args"].get("acq_func", "EI")
        acq_func_kwargs = res.specs["args"].get("acq_func_kwargs", {})

    if acq_func_kwargs is None:
        acq_func_kwargs = {}
    if acq_func is None or acq_func == "gp_hedge":
        acq_func = "EI"
    if n_random is None:
        n_random = len(res.x_iters) - len(res.models)

    # if objective is not None:
    #     fx = np.array([objective(x_i) for x_i in x])
    if n_calls < 0:
        model = res.models[-1]
        curr_x_iters = res.x_iters
        curr_func_vals = res.func_vals
    else:
        model = res.models[n_calls]

        curr_x_iters = res.x_iters[:n_random + n_calls]
        curr_func_vals = res.func_vals[:n_random + n_calls]

    # Plot true function
    # df = pd.read_json('./')

    # Plot true function.
    # if objective is not None:
    #     ax.plot(x, fx, "r--", label="True (unknown)")
    #     ax.fill(np.concatenate(
    #         [x, x[::-1]]),
    #         np.concatenate(([fx_i - 1.9600 * noise_level
    #                          for fx_i in fx],
    #                         [fx_i + 1.9600 * noise_level
    #                          for fx_i in fx[::-1]])),
    #         alpha=.2, fc="r", ec="None")

    # Plot GP(x) + contours
    if show_mu:
        y_pred, sigma = model.predict(x_model, return_std=True)
        ax.plot(x, y_pred, "g--", label=r"$\mu_{GP}(x)$")
        ax.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([y_pred - 1.9600 * sigma,
                                (y_pred + 1.9600 * sigma)[::-1]]),
                alpha=.2, fc="g", ec="None")

    # Plot sampled points
    if show_observations:
        ax.plot(curr_x_iters, curr_func_vals,
                "r.", markersize=8, label="Observations")
    if (show_mu or show_observations or objective is not None)\
            and show_acq_func:
        ax_ei = ax.twinx()
        ax_ei.set_ylabel(str(acq_func) + "(x)")
        plot_both = True
    else:
        ax_ei = ax
        plot_both = False
    if show_acq_func:
        acq = _gaussian_acquisition(x_model, model,
                                    y_opt=np.min(curr_func_vals),
                                    acq_func=acq_func,
                                    acq_func_kwargs=acq_func_kwargs)
        next_x = x[np.argmin(acq)]
        next_acq = acq[np.argmin(acq)]
        acq = - acq
        next_acq = -next_acq
        ax_ei.plot(x, acq, "b", label=str(acq_func) + "(x)")
        if not plot_both:
            ax_ei.fill_between(x.ravel(), 0, acq.ravel(),
                               alpha=0.3, color='blue')

        if show_next_point and next_x is not None:
            ax_ei.plot(next_x, next_acq, "bo", markersize=6,
                       label="Next query point")

    if show_title:
        ax.set_title(r"x* = %.4f, f(x*) = %.4f" % (res.x[0], res.fun))
    # Adjust plot layout
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    if show_legend:
        if plot_both:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_ei.get_legend_handles_labels()
            ax_ei.legend(lines + lines2, labels + labels2, loc="best",
                         prop={'size': 6}, numpoints=1)
        else:
            ax.legend(loc="best", prop={'size': 6}, numpoints=1)

    return ax


def _evenly_sample(dim, n_points):
    """Return `n_points` evenly spaced points from a Dimension.

    Parameters
    ----------
    dim : `Dimension`
        The Dimension to sample from.  Can be categorical; evenly-spaced
        category indices are chosen in order without replacement (result
        may be smaller than `n_points`).

    n_points : int
        The number of points to sample from `dim`.

    Returns
    -------
    xi : np.array
        The sampled points in the Dimension.  For Categorical
        dimensions, returns the index of the value in
        `dim.categories`.

    xi_transformed : np.array
        The transformed values of `xi`, for feeding to a model.
    """
    cats = np.array(getattr(dim, 'categories', []), dtype=object)
    if len(cats):  # Sample categoricals while maintaining order
        xi = np.linspace(0, len(cats) - 1, min(len(cats), n_points),
                         dtype=int)
        xi_transformed = dim.transform(cats[xi])
    else:
        bounds = dim.bounds
        # XXX use linspace(*bounds, n_points) after python2 support ends
        xi = np.linspace(bounds[0], bounds[1], n_points)
        xi_transformed = dim.transform(xi)
    return xi, xi_transformed


def _gaussian_acquisition(X, model, y_opt=None, acq_func="LCB",
                          return_grad=False, acq_func_kwargs=None):
    """
    Wrapper so that the output of this function can be
    directly passed to a minimizer.
    """
    # Check inputs
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X is {}-dimensional, however,"
                         " it must be 2-dimensional.".format(X.ndim))

    if acq_func_kwargs is None:
        acq_func_kwargs = dict()
    xi = acq_func_kwargs.get("xi", 0.01)
    kappa = acq_func_kwargs.get("kappa", 1.96)

    # Evaluate acquisition function
    per_second = acq_func.endswith("ps")
    if per_second:
        model, time_model = model.estimators_

    if acq_func == "LCB":
        func_and_grad = gaussian_lcb(X, model, kappa, return_grad)
        if return_grad:
            acq_vals, acq_grad = func_and_grad
        else:
            acq_vals = func_and_grad

    elif acq_func in ["EI", "PI", "EIps", "PIps"]:
        if acq_func in ["EI", "EIps"]:
            func_and_grad = gaussian_ei(X, model, y_opt, xi, return_grad)
        else:
            func_and_grad = gaussian_pi(X, model, y_opt, xi, return_grad)

        if return_grad:
            acq_vals = -func_and_grad[0]
            acq_grad = -func_and_grad[1]
        else:
            acq_vals = -func_and_grad

        if acq_func in ["EIps", "PIps"]:

            if return_grad:
                mu, std, mu_grad, std_grad = time_model.predict(
                    X, return_std=True, return_mean_grad=True,
                    return_std_grad=True)
            else:
                mu, std = time_model.predict(X, return_std=True)

            # acq = acq / E(t)
            inv_t = np.exp(-mu + 0.5*std**2)
            acq_vals *= inv_t

            # grad = d(acq_func) * inv_t + (acq_vals *d(inv_t))
            # inv_t = exp(g)
            # d(inv_t) = inv_t * grad(g)
            # d(inv_t) = inv_t * (-mu_grad + std * std_grad)
            if return_grad:
                acq_grad *= inv_t
                acq_grad += acq_vals * (-mu_grad + std*std_grad)

    else:
        raise ValueError("Acquisition function not implemented.")

    if return_grad:
        return acq_vals, acq_grad
    return acq_vals