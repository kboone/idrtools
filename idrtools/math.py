import numpy as np
import iminuit


def unbiased_std(x):
    x = np.asarray(x) - np.mean(x)
    return np.sqrt(np.sum(x*x) / (len(x) - 1))


def nmad(x):
    return 1.4826 * np.median(np.abs(np.asarray(x) - np.median(x)))


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
    window_size = int(len(x) * window_frac)
    out = np.zeros(len(x))
    for i in range(len(x)):
        min_index = i - window_size / 2.
        if min_index < 0:
            min_index = 0

        max_index = i + window_size / 2.
        if max_index > len(x):
            max_index = len(x)

        out[i] = func(x[min_index:max_index])

    return out


def windowed_nmad(x, window_frac=0.05):
    return apply_windowed_function(x, nmad, window_frac)


def windowed_rms(x, window_frac=0.05):
    return apply_windowed_function(x, rms, window_frac)


def fit_global_values(id_1, id_2, diffs, return_fitted_diffs=False):
    """Perform a global fit on a set of difference values from a datset."""
    all_vars = np.unique([id_1, id_2])
    var_dict = {var: i for i, var in enumerate(all_vars)}

    def global_fit_dist(*vals):
        # vals are all of the values *except* the first. The first is
        # set so that the sum of all values is 0. We would have a
        # reduncancy otherwise.
        full_vals = [-np.sum(vals)] + list(vals)

        id_1_vals = np.array([full_vals[var_dict[var]] for var in id_1])
        id_2_vals = np.array([full_vals[var_dict[var]] for var in id_2])
        global_fit_vals = id_1_vals - id_2_vals

        chisq = ((diffs - global_fit_vals)**2).sum()

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
