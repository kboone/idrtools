import numpy as np
import iminuit
from functools import partial
from scipy.linalg import pinvh, LinAlgError
from scipy.stats import binned_statistic


class IdrToolsMathException(Exception):
    pass


def nmad(x, *args, unbiased=False, centered=False, **kwargs):
    x = np.asarray(x)
    if not centered:
        x = x - np.median(x, *args, **kwargs)

    nmad = 1.4826 * np.median(np.abs(x), *args, **kwargs)

    if unbiased:
        nmad = nmad * x.size / (x.size - 1)

    return nmad


def nmad_centered(x, *args, **kwargs):
    return nmad(x, *args, centered=True, **kwargs)


def rms(x):
    x = np.asarray(x)
    return np.sqrt(np.sum(x*x) / x.size)


def cum_rms(x):
    x = np.asarray(x)
    num = np.cumsum(x**2)
    denom = np.arange(x.size) + 1
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

def plot_binned_nmad_centered(x, y, *args, **kwargs):
    return plot_binned_function(x, y, nmad_centered, *args,
                                uncertainties=False, **kwargs)

def plot_binned_rms(x, y, *args, **kwargs):
    return plot_binned_function(x, y, rms, *args, uncertainties=False,
                                **kwargs)

def plot_binned_std(x, y, *args, **kwargs):
    return plot_binned_function(x, y, np.std, *args, uncertainties=False,
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


def _apply_bootstrap_mask(vals, mask):
    if len(np.shape(vals)) != 1 or len(vals) != mask.shape[0]:
        # Not something to apply the bootstrap to
        return vals
    
    else:
        return vals[mask]


def bootstrap_statistic(statistic, vals, *args, num_resamples=10000, **kwargs):
    """Use bootstrap resampling to estimate the uncertainty on a statistic.

    This supports passing arbitrary arguments to the called statistic function.
    If those arguments have the same shape as vals, they will be bootstrapped
    too. Otherwise, they will be passed as-is.
    """
    stat = statistic(vals, *args, **kwargs)
    
    if len(np.shape(vals)) != 1:
        raise Exception("bootstrap_statistic only supported for 1-D inputs.")
        
    bootstrap_idx = np.random.choice(len(vals), (len(vals), num_resamples))

    bootstrap_vals = vals[bootstrap_idx]
    
    bootstrap_args = []
    for arg in args:
        bootstrap_args.append(_apply_bootstrap_mask(arg, bootstrap_idx))
        
    bootstrap_kwargs = {}
    for arg_name, arg in kwargs.items():
        bootstrap_kwargs[arg_name] = _apply_bootstrap_mask(arg, bootstrap_idx)
        
    bootstrap_stat = statistic(bootstrap_vals, *bootstrap_args, axis=0,
                               **bootstrap_kwargs)
    stat_err = np.std(bootstrap_stat, axis=-1)
    
    if len(np.shape(stat_err)) > 0:
        stat = np.array(stat)
    
    return stat, stat_err


def hessian_to_covariance(hessian):
    """Safely invert a Hessian matrix to get a covariance matrix.

    Sometimes, parameters can have wildly different scales from each other. What we
    actually care about is having the same relative precision on the error of each
    parameter rather than the absolute precision. In that case, we can normalize the
    Hessian prior to inverting it, and then renormalize afterwards. This deals with the
    problem of varying scales of parameters gracefully.
    """
    # Choose scales to set the diagonal of the hessian to 1.
    scales = np.sqrt(np.diag(hessian))
    norm_hessian = hessian / np.outer(scales, scales)

    # Now invert the scaled Hessian using a safe inversion algorithm
    inv_norm_hessian = pinvh(norm_hessian)

    # Add the scales back in.
    covariance = inv_norm_hessian / np.outer(scales, scales)

    return covariance


def calculate_covariance_finite_difference(negative_log_likelihood, parameter_names,
                                           values, bounds, verbose=False):
    """Estimate the covariance of the parameters of negative log likelihood function
    numerically.

    We do a 2nd order finite difference estimate of the covariance matrix.
    For this, the formula is:
    d^2f(dx1*dx2) = ((f(x+e1+e2) - f(x+e1-e2) - f(x-e1+e2) + f(x-e1-e2))
                     / 4*e1*e2)
    So we need to calculate all the f(x +/-e1 +/-e2) terms (where e1 and
    e2 are small steps in 2 possibly different directions).

    We use adaptive step sizes to build a robust estimate of the Hessian, and invert it
    to obtain the covariance matrix.

    Parameters
    ----------
    negative_log_likelihood : function
        Negative log-likelihood function to evaluate. This should take as input a list
        of parameter values.
    parameter_names : list of str
        Names of each of the parameters.
    values : list of floats
        Values of each of the parameters at the maximum likelihood location. The
        likelihood should be run through a minimizer before calling this function, and
        the resulting values should be passed into this function.
    bounds : list of tuples
        Bounds for each parameter. For each parameter, this should be a two parameter
        tuple with the first entry being the minimum bound and the second entry being
        the maximum bound.
    verbose : bool
        If True, output diagnostic messages. Default: False
    """
    # The three terms here are the corresponding weight, the sign of e1 and
    # the sign of e2.
    difference_info = [
        (+1/4., +1., +1.),
        (-1/4., +1., -1.),
        (-1/4., -1., +1.),
        (+1/4., -1., -1.),
    ]

    num_variables = len(parameter_names)

    # Determine good step sizes. Since we have a chi-square function, a 1-sigma
    # change in a parameter corresponds to a 1 unit change in the output
    # chi-square function. We want our steps to change the chi-square function
    # by an amount of roughly 1e-5 (far from machine precision, but small
    # enough to be probing locally). We start by guessing a step size of 1e-5
    # (which is typically pretty reasonable for parameters that are of order 1)
    # and then bisect to find the right value.
    steps = []
    ref_likelihood = negative_log_likelihood(values)

    for parameter_idx in range(len(parameter_names)):
        step = 1e-5
        min_step = None
        max_step = None

        # Move away from the nearest bounds to avoid boundary issues.
        min_bound, max_bound = bounds[parameter_idx]
        value = values[parameter_idx]
        if min_bound is None:
            if max_bound is None:
                # No bounds, doesn't matter what we pick.
                direction = +1.
            else:
                # Max bound only
                direction = -1.
        else:
            if max_bound is None:
                # Min bound only
                direction = +1.
            else:
                # Both bounds, move away from the nearest bound.
                if value - min_bound > max_bound - value:
                    direction = -1.
                else:
                    direction = 1.

        while True:
            # Estimate the second derivative numerator for a finite difference
            # calculation. We want to choose a step size that sets this to a
            # reasonable value. Note that we move only in the direction away
            # from the nearest boundary, so this isn't centered at the correct
            # position, but this is only to get an initial estimate of the
            # scale so it doesn't matter.
            step_values = values.copy()
            step_values[parameter_idx] += step * direction
            step_1_likelihood = negative_log_likelihood(step_values)
            step_values[parameter_idx] += step * direction
            step_2_likelihood = negative_log_likelihood(step_values)
            diff = (
                0.25 * step_2_likelihood
                - 0.5 * step_1_likelihood
                + 0.25 * ref_likelihood
            )

            if diff < -1e-4:
                # We found a minimum that is better than the supposed true
                # minimum. This indicates that something is wrong because the
                # minimizer failed.
                raise IdrToolsMathException(
                    "Second derivative is negative when varying %s to "
                    "calculate covariance matrix! Something is very wrong! "
                    "(step=%f, second derivative=%f)" %
                    (parameter_names[parameter_idx], step, diff)
                )

            if diff < 1e-6:
                # Too small step size, increase it.
                min_step = step
                if max_step is not None:
                    step = (step + max_step) / 2.
                else:
                    step = step * 2.
            elif diff > 1e-4:
                # Too large step size, decrease it.
                max_step = step
                if min_step is not None:
                    step = (step + min_step) / 2.
                else:
                    step = step / 2.
            elif step > 1e9:
                # Shouldn't need steps this large. This only happens if one
                # parameter doesn't affect the model at all, in which case we
                # can't calculate the covariance.
                raise IdrToolsMathException(
                    "Parameter %s doesn't appear to affect the model! Cannot "
                    "estimate the covariance." % parameter_names[parameter_idx]
                )
            else:
                # Good step size, we're done.
                break

        steps.append(step)

    steps = np.array(steps)
    if verbose:
        print("Finite difference covariance step sizes: %s" % steps)

    difference_matrices = []

    # If we are too close to a boundary, shift the center position by the step
    # size. This isn't technically right, but we can't evaluate past bounds or
    # we run into major issues with parameters that have real physical bounds
    # (eg: airmass below 1.)
    original_values = values
    values = original_values.copy()
    for parameter_idx in range(len(parameter_names)):
        name = parameter_names[parameter_idx]
        min_bound, max_bound = bounds[parameter_idx]
        value = values[parameter_idx]
        step = steps[parameter_idx]

        direction_str = None
        if min_bound is not None and value - 2*step < min_bound:
            direction_str = "up"
            values[parameter_idx] = min_bound + 2*step
        elif max_bound is not None and value + 2*step > max_bound:
            direction_str = "down"
            values[parameter_idx] = max_bound - 2*step

        if direction_str is not None and verbose:
            print("WARNING: Parameter %s is at bound! Moving %s by %g to "
                  "calculate covariance!" % (name, direction_str, 2*step))

    # Calculate all of the terms that will be required to calculate the finite
    # differences. Note that there is a lot of reuse of terms, so here we
    # calculate everything that is needed and build a set of matrices for each
    # step combination.
    for weight, sign_e1, sign_e2 in difference_info:
        matrix = np.zeros((num_variables, num_variables))
        for i in range(num_variables):
            for j in range(num_variables):
                if i > j:
                    # Symmetric
                    continue

                step_values = values.copy()
                step_values[i] += sign_e1 * steps[i]
                step_values[j] += sign_e2 * steps[j]
                likelihood = negative_log_likelihood(step_values)
                matrix[i, j] = likelihood
                matrix[j, i] = likelihood

        difference_matrices.append(matrix)

    # Hessian
    hessian = np.zeros((num_variables, num_variables))
    for i in range(num_variables):
        for j in range(num_variables):
            if i > j:
                continue

            val = 0.

            for (weight, sign_e1, sign_e2), matrix in \
                    zip(difference_info, difference_matrices):
                val += weight * matrix[i, j]

            val /= steps[i] * steps[j]

            hessian[i, j] = val
            hessian[j, i] = val

    # Invert the Hessian to get the covariance matrix
    try:
        cov = hessian_to_covariance(hessian)
    except LinAlgError:
        raise IdrToolsMathException("Covariance matrix is not well defined!")

    variance = np.diag(cov)

    if np.any(variance < 0):
        raise IdrToolsMathException("Covariance matrix is not well defined! "
                                    "Found negative variances.")

    return cov
