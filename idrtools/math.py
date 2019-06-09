import numpy as np
import iminuit
from functools import partial
from scipy.stats import binned_statistic


class IdrToolsMathException(Exception):
    pass


def wilson_score(n, c, z):
    """Calculate the Wilson score
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    Parameters
    ==========
    n : int
        The total number of samples.
    c : int
        The total number of positive samples.
    z : float
        The number of standard deviations to calculate the Wilson score for
        (can be positive or negative).
    """
    p = c / n
    return (p + z*z/(2*n) + z*np.sqrt((p*(1-p)+z*z/(4*n))/n)) / (1+z*z/n)


def unbiased_std(x):
    x = np.asarray(x) - np.mean(x)
    return np.sqrt(np.sum(x*x) / (len(x) - 1))


def nmad(x, *args, **kwargs):
    return 1.4826 * np.median(
        np.abs(np.asarray(x) - np.median(x, *args, **kwargs)),
        *args, **kwargs
    )


def nmad2(x, *args, **kwargs):
    """NMAD without median subtraction"""
    return 1.4826 * np.median(
        np.abs(np.asarray(x)),
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


def plot_binned_function(x, y, func, bins=10, equal_bin_counts=False,
                         uncertainties=True, mode='step', **kwargs):
    """Plot the given function applied to the y variable, grouped by bins of
    the x variable.

    - bins and range are passed to scipy.stats.binned_statistic, and can be
    used in several ways as described in that function's documentation. The
    simplest usage is for bins to be an integer that specifies how many
    equal-width bins to have, and for range to be a tuple that specifies the
    edges of the domain to include.
    - If equal_bin_counts is True, then bins are created that have the same
    numbers of counts. Otherwise, bins are created to have the same widths in
    the x-variable. When equal_bin_counts is True, then bins must be an
    integer.

    There are several modes available for plotting, specified with the mode
    keyword.
    - step: draw a flat line for each group that spans the full width of the
    group.
    - hline: draw disconnected horizontal lines for each the bins.
    - scatter: do a scatter plot of the results.
    - plot: draw lines between the bins, each of which is a single point.

    If uncertainties is True, plot uncertainties for each bin. This is only
    currently available for a subset of functions where this is somewhat
    well-defined, and uncertainties are estimated from the data.
    """

    from matplotlib import pyplot as plt

    x = np.ravel(x)
    y = np.ravel(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # range is the keyword used by scipy.stats.binned_function, but it is also
    # a python keyword that we don't want to overwrite. Ugh. Internally, we use
    # the term "use_range", but use the variable "range" as the input to this
    # function for consistency.
    use_range = kwargs.pop(range, None)

    bin_kwargs = {
        'bins': bins,
        'range': use_range,
    }

    if equal_bin_counts:
        # Figure out the percentiles at which we should split the bins.
        if use_range is not None:
            use_x = x[(x > use_range[0]) & (x < use_range[1])]
        else:
            use_x = x

        use_x = np.sort(use_x)

        bin_edges = [use_x[0]]

        last_idx = 0
        warned = False

        for i in range(1, bins):
            next_idx = i * len(use_x) // bins

            # Edge case: if we specify more bins than there are points, bins
            # will get duplicated. Handle this (somewhat) gracefully by just
            # not adding more bins.
            if next_idx == last_idx or next_idx == len(use_x):
                if not warned:
                    print("Warning! Specified too many bins in "
                          "plot_binned_function for equal bin counts. "
                          "Dropping some bins!")
                    warned = True
                continue

            last_idx = next_idx

            next_edge = (use_x[next_idx + 1] + use_x[next_idx]) / 2.

            # Other edge case: if there is a pileup at a certain x-value, then
            # we can run into issues with putting multiple bin edges at the
            # same point. Just don't put a bin edge there.
            if next_edge == bin_edges[-1]:
                if not warned:
                    print("Warning! Pileup of x-values at %f in "
                          "plot_binned_function for equal bin counts. "
                          "Dropping some bins!" % next_edge)
                    warned = True
                continue

            bin_edges.append(next_edge)

        last_bin = use_x[-1]
        if last_bin == bin_edges[-1]:
            if not warned:
                print("Warning! Pileup of x-values at %f in "
                      "plot_binned_function for equal bin counts. "
                      "Dropping some bins!" % next_edge)
                warned = True
        else:
            bin_edges.append(last_bin)

        bin_kwargs['bins'] = bin_edges

    # Estimate the statistic
    use_func = func

    if func == 'binomial':
        use_func = 'mean'

    statistic, bin_edges, binnumber = binned_statistic(x, y, use_func,
                                                       **bin_kwargs)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    if mode == 'scatter':
        plot = plt.scatter(bin_centers, statistic, **kwargs)
    elif mode == 'plot':
        # Cut out NaNs which will put holes in the plot otherwise.
        mask = np.isfinite(statistic)
        plot = plt.plot(bin_centers[mask], statistic[mask], **kwargs)
    elif mode == 'step':
        plot = plt.step(bin_edges, np.hstack([statistic, statistic[-1]]),
                        where='post', **kwargs)
    elif mode == 'hline':
        plot = plt.hlines(statistic, bin_edges[:-1], bin_edges[1:], **kwargs)
    else:
        raise IdrToolsMathException("Unknown mode %s!" % mode)

    if uncertainties:
        # Define an estimator of the uncertainty on the statistic. For each
        # estimator, we define `calc_uncertainty` which should evaluate the
        # uncertainty on a set of numbers. By default, we assume that the 
        # estimators are symmetric. If that is not the case, then
        # `calc_uncertainty_negative` can be defined. In that case,
        # the result of `calc_uncertainty` is interpreted as the positive
        # uncertainty, and the result of `calc_uncertainty_negative` is
        # interpreted as the negative uncertainty.
        calc_uncertainty_negative = None
        if func == 'median':
            def calc_uncertainty(bin_vals):
                denom = max(len(bin_vals), 2)
                return nmad(bin_vals) / np.sqrt(denom - 1)
        elif func == 'mean':
            def calc_uncertainty(bin_vals):
                denom = max(len(bin_vals), 2)
                return np.std(bin_vals) / np.sqrt(denom - 1)
        elif func == 'binomial':
            # Use the Wilson score interval to estimate the uncertainty
            # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
            def calc_uncertainty(bin_vals, z=1):
                c = np.sum(bin_vals)
                n = len(bin_vals)

                return np.abs(wilson_score(n, c, z) - c / n)
            calc_uncertainty_negative = partial(calc_uncertainty, z=-1)
        else:
            raise IdrToolsMathException(
                "Can't do errors for given function! (%s)" % func
            )

        bin_uncertainties, uncertainty_bin_edges, uncertainty_binnumber = \
            binned_statistic(x, y, calc_uncertainty, **bin_kwargs)
        assert np.all(uncertainty_binnumber == binnumber)

        if calc_uncertainty_negative is not None:
            # Have uncertainties defined for both directions.
            neg_bin_uncertainties, neg_bin_edges, neg_binnumber = \
                binned_statistic(x, y, calc_uncertainty_negative, **bin_kwargs)
            assert np.all(neg_binnumber == binnumber)

            bin_uncertainties = (neg_bin_uncertainties, bin_uncertainties)

        uncertainty_kwargs = kwargs.copy()

        if 'fmt' not in uncertainty_kwargs:
            uncertainty_kwargs['fmt'] = 'none'

        # Copy the color from the main plot
        uncertainty_kwargs['c'] = plot[0].get_color()

        # Don't label the uncertainty
        uncertainty_kwargs['label'] = ''

        plt.errorbar(bin_centers, statistic, yerr=bin_uncertainties,
                     **uncertainty_kwargs)


    return bin_centers, statistic


def plot_binned_nmad(x, y, *args, **kwargs):
    return plot_binned_function(x, y, nmad, *args, uncertainties=False,
                                **kwargs)


def plot_binned_rms(x, y, *args, **kwargs):
    return plot_binned_function(x, y, rms, *args, uncertainties=False,
                                **kwargs)


def plot_binned_median(x, y, *args, **kwargs):
    return plot_binned_function(x, y, 'median', *args, **kwargs)


def plot_binned_mean(x, y, *args, **kwargs):
    return plot_binned_function(x, y, 'mean', *args, **kwargs)


def plot_binned_binomial(x, y, *args, **kwargs):
    """Plot statistics for a variable that is either 0 or 1.

    This will provide an estimate of the binomial probability p for each bin,
    and the uncertainty on the estimate of p can also be calculated.
    """
    return plot_binned_function(x, y, 'binomial', *args, **kwargs)


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
