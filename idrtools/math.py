import numpy as np
import iminuit
from scipy.stats import binned_statistic


def unbiased_std(x):
    x = np.asarray(x) - np.mean(x)
    return np.sqrt(np.sum(x*x) / (len(x) - 1))


def nmad(x, *args, **kwargs):
    return 1.4826 * np.median(
        np.abs(np.asarray(x) - np.median(x, *args, **kwargs)),
        *args, **kwargs
    )


def rms(x):
    x = np.asarray(x)
    return np.sqrt(np.sum(x*x) / len(x))


def cum_rms(x):
    x = np.asarray(x)
    num = np.cumsum(x**2)
    denom = np.arange(len(x)) + 1
    return np.sqrt(num / denom)


def cum_nmad(x):
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = nmad(x[:i+1])

    return out


def apply_windowed_function(x, func, window_frac=0.05):
    x = np.asarray(x)

    window_size = int(np.around(len(x) * window_frac / 2.))
    out = np.zeros(len(x))
    for i in range(len(x)):
        min_index = i - window_size
        if min_index < 0:
            min_index = 0

        max_index = i + window_size
        if max_index > len(x):
            max_index = len(x)

        out[i] = func(x[min_index:max_index])

    return out


def windowed_nmad(x, *args, **kwargs):
    return apply_windowed_function(x, nmad, *args, **kwargs)


def windowed_rms(x, *args, **kwargs):
    return apply_windowed_function(x, rms, *args, **kwargs)


def windowed_median(x, *args, **kwargs):
    return apply_windowed_function(x, np.median, *args, **kwargs)


def windowed_mean(x, *args, **kwargs):
    return apply_windowed_function(x, np.mean, *args, **kwargs)


def plot_windowed_function(x, y, func, window_frac=0.05, *args, **kwargs):
    from matplotlib import pyplot as plt

    x = np.asarray(x)
    y = np.asarray(y)

    order = np.argsort(x)
    x_ordered = x[order]
    y_ordered = y[order]

    result = apply_windowed_function(y_ordered, func, window_frac)

    plt.plot(x_ordered, result, *args, **kwargs)


def plot_windowed_nmad(x, y, *args, **kwargs):
    return plot_windowed_function(x, y, nmad, *args, **kwargs)


def plot_windowed_rms(x, y, *args, **kwargs):
    return plot_windowed_function(x, y, rms, *args, **kwargs)


def plot_windowed_median(x, y, *args, **kwargs):
    return plot_windowed_function(x, y, np.median, *args, **kwargs)


def plot_windowed_mean(x, y, *args, **kwargs):
    return plot_windowed_function(x, y, np.mean, *args, **kwargs)


def plot_binned_function(x, y, func, *args, scatter=False, **kwargs):
    from matplotlib import pyplot as plt

    x = np.asarray(x)
    y = np.asarray(y)

    bin_kwargs = {}
    if 'bins' in kwargs:
        bin_kwargs['bins'] = kwargs.pop('bins')
    if 'range' in kwargs:
        bin_kwargs['range'] = kwargs.pop('range')

    statistic, bin_edges, binnumber = binned_statistic(x, y, func,
                                                       **bin_kwargs)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    if scatter:
        plt.scatter(bin_centers, statistic, *args, **kwargs)
    else:
        plt.plot(bin_centers, statistic, *args, **kwargs)


def plot_binned_nmad(x, y, *args, **kwargs):
    return plot_binned_function(x, y, nmad, *args, **kwargs)


def plot_binned_rms(x, y, *args, **kwargs):
    return plot_binned_function(x, y, rms, *args, **kwargs)


def plot_binned_median(x, y, *args, **kwargs):
    return plot_binned_function(x, y, 'median', *args, **kwargs)


def plot_binned_mean(x, y, *args, **kwargs):
    return plot_binned_function(x, y, 'mean', *args, **kwargs)


def fit_global_values(id_1, id_2, diffs, weights=None,
                      return_fitted_diffs=False):
    """Perform a global fit on a set of difference values from a datset.

    id_1 and id_2 should be a list of the objects in each of the pairs. diffs
    is a set of differences to target. weights gives the weights to use for
    each pairing in the fit.
    """
    all_vars = np.unique(np.concatenate([id_1, id_2]))
    var_dict = {var: i for i, var in enumerate(all_vars)}

    def global_fit_dist(*vals):
        # vals are all of the values *except* the first. The first is
        # set so that the sum of all values is 0. We would have a
        # reduncancy otherwise.
        full_vals = [-np.sum(vals)] + list(vals)

        id_1_vals = np.array([full_vals[var_dict[var]] for var in id_1])
        id_2_vals = np.array([full_vals[var_dict[var]] for var in id_2])
        global_fit_vals = id_1_vals - id_2_vals

        fit_errs = (diffs - global_fit_vals)**2

        if weights is not None:
            fit_errs *= weights

        chisq = fit_errs.sum()

        return chisq

    error_guess = np.std(diffs) / np.sqrt(2)

    param_names = ['m' + str(i) for i in range(1, len(all_vars))]
    params = {i: 0. for i in param_names}
    params.update({"error_" + i: error_guess for i in param_names})
    minuit = iminuit.Minuit(
        global_fit_dist,
        errordef=1.,
        forced_parameters=param_names,
        print_level=0,
        **params
    )
    migrad_result = minuit.migrad()

    if not migrad_result[0].is_valid:
        print("-"*80)
        print("ERROR: Global fit failed to converge!")
        print(migrad_result)
        print("-"*80)
        return [0.] * len(all_vars)

    fitted_vals = [minuit.values[minuit.pos2var[i]] for i in
                   range(len(all_vars)-1)]
    fitted_vals = np.array([-np.sum(fitted_vals)] + fitted_vals)

    if return_fitted_diffs:
        id_1_vals = np.array([fitted_vals[var_dict[var]] for var in id_1])
        id_2_vals = np.array([fitted_vals[var_dict[var]] for var in id_2])
        global_fit_vals = id_1_vals - id_2_vals
        return (all_vars, fitted_vals, global_fit_vals)

    return all_vars, fitted_vals
