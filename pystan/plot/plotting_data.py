import logging
logger = logging.getLogger('pystan')

import numpy as np
from pystan._compat import string_types

def hist_data(vec, limits=None, nbins=None):
    """Function to calculate data for histogram (heights and edges)
    Uses np.histogram.

    Parameters
    ----------
    vec : ndarray
    limits: tuple, optional
        Left and right limits for the bins. Default is min-max.
    nbins : int, optional
        Number of bins used in the calculation. Minimum bin width equals 1
        due to specialization for integer data. Default is `max(10, min(n//10, 100))`.

    Returns
    -------
    hist : ndarray
        Density values for the bins.
    edges : ndarray
        Bin edges.
    """
    n = len(vec)
    if limits is not None:
        vmin, vmax = limits
    else:
        vmin, vmax = vec.min(), vec.max()
    if nbins is None:
        nbins = max(10, min(n//10, 100))
    bins = range(vmin, vmax + 2, max((vmax-vmin)// nbins, 1))
    hist, edges = np.histogram(vec, bins=bins, density=True)
    return hist, edges

def gaussian(n, sig):
    """Gaussian kernel function

    Parameters
    ----------
    n : int
        Number of kernel points.
    sig : float
        Width of the kernel.

    Returns
    -------
    gaussian : ndarray
    """
    N = np.arange(0, n) - (n - 1.0) / 2
    lg = -N ** 2 / ( 2 * sig ** 2 )
    return np.exp(lg)

def fftconvolve(grid, kernel):
    """Convolve kernel over grid with fft trick

    Parameters
    ----------
    grid : ndarray
    kernel : ndarray

    Returns
    -------
    grid : ndarray
        Grid convolved with the kernel.
    """
    shape1 = grid.shape[0]
    shape2 = kernel.shape[0]
    shape = int(shape1 + shape2 - 1)
    fft_shape = 2 ** shape.bit_length()
    fft_slice = slice(0, shape)
    fft_grid = np.fft.rfft(grid, fft_shape)
    fft_kernel = np.fft.rfft(kernel, fft_shape)
    fft_product = fft_grid * fft_kernel
    convolved_grid = np.fft.irfft(fft_product, fft_shape)[fft_slice]
    return convolved_grid

def kde_data(vec, limits=None, c=1):
    """Function to calculate approximate kernel density estimation data with scotts_factor.
    Speeds up the computation with fft trick.

    The code is an adaption from https://github.com/mfouesneau/faststats

    Parameters
    ----------
    vec : ndarray
    limits : tuple, optional
        Left and right limits for the bins (fft trick).
    c : float, optional
        Constant for the scotts factor: `n ** (-0.2) * c`.

    Returns
    -------
    bins : ndarray
        Positions for the kde estimates.
    grid : ndarray
        Kde estimates.
    """
    n = len(vec)
    nx = 200
    scotts_factor = nx ** (-0.2) * c
    if limits is not None:
        vmin, vmax = limits
        if vmin < vmin.min() or vmax > vmax.max():
            logger.warning("Points outside the limit range could create biased density estimate")
        vec = vec[ (vec >= vmin) & (vec <= vmax) ]
    else:
        vmin, vmax = vec.min(), vec.max()
    bins = np.linspace(vmin, vmax, nx)
    dx = (vmax-vmin) / (nx-1)
    xyi = np.digitize(vec, bins)
    std_x = np.std(xyi)
    kern_nx = int(scotts_factor * 2 * np.pi * std_x)
    fft_nx = kern_nx * 3 + kern_nx % 1
    kernel = gaussian(fft_nx, scotts_factor*std_x)
    npad = np.min((nx, 2*kern_nx))
    grid, _ = np.histogram(vec, bins=bins)
    grid = np.r_[grid[npad:0:-1], grid, grid[nx:nx-npad:-1]]
    grid = fftconvolve(grid, kernel)
    grid_slice = slice(npad+fft_nx//2-1, nx+npad+fft_nx//2-1)
    grid = grid[grid_slice]
    norm_factor = n * dx * ( 2 * np.pi * std_x ** 2 * scotts_factor ** 2 ) ** 0.5
    grid = grid / norm_factor
    return bins, grid

def traceplot_data(fit, pars, dtypes=None, density=True, split_pars=False, **kwargs):
    """Function to yield data for the traceplot.

    Parameters
    ----------
    fit : StanFit4Model object
    pars : tuple
        Parameters plotted.
    dtypes : dict, optional
        Dictionary containing parameters as a key value and transform-function value.
    density : bool, optional, default True
        If False plot only the trace data, exclude the density plot
    split_pars : bool, optional, default False
        If True plot each parameter including vector and matrix components in their own axis
    c_kde : int, optional
        Constant for the kde function, ignored if density==False
    nbins : int, optional
        Maximum number of bins for histogram function, ignored if density is False.
    inc_warmup : bool, optional, default False
        Include warmup

    Yields
    ------
    data : dict
        dict contains
            par, parameter name
            name, parameter component name
            vec, raw data
            hist, histogram data, optional
            kde, kde data, optional
    """
    # TODO: Add option to plot chains independently
    c_kde = kwargs.get('c_kde', 1)
    nbins = kwargs.get('nbins', None)
    inc_warmup = kwargs.get('inc_warmup', False)
    sampler_params = {'accept_stat__', 'stepsize__', 'treedepth__', \
                      'n_leapfrog__', 'divergent__', 'energy__'}
    for par in pars:
        if par in sampler_params:
            n_save = fit.sim['n_save'][0]
            if inc_warmup:
                n_save = n_save - fit.sim['warmup']
            vecs = np.concatenate([sp['par'][-n_save:] for sp in fit.get_sampler_params()])
        else:
            vecs = fit.extract(pars=par, permuted=False, dtypes=dtypes, inc_warmup=inc_warmup)[par]
        nchains = fit.sim['chains']
        if nchains == 1:
            if len(vecs.shape) == 1:
                # add chain dimension
                vecs = np.expand_dims(vecs, -1)
        par_dims = vecs.shape[2:]
        if not par_dims:
            # add parameter dimension
            vecs = np.expand_dims(vecs, -1)
            par_dims = vecs.shape[2:]
        # reshape and translate for plotting
        vecs = vecs.reshape([vecs.shape[0]*vecs.shape[1]]+list(par_dims), order='F').T
        # limits as min-max for each vec
        limits = None
        m = np.multiply.reduce(par_dims)
        indices = np.c_[np.unravel_index(np.arange(m), par_dims, order='F')]
        # This loop creates name instead of `fit.flatnames`.
        for idx in indices:
            vec = vecs[tuple(idx)]
            is_unique = len(np.unique(vec)) == 1
            # use par if value scalar
            if par_dims == (1,):
                name = par
            else:
                # add 1 for the index
                name = "{}[{}]".format(par, ",".join(map(str, idx+1)))
            if density:
                if isinstance(vec[0], (int, np.integer, np.uint)):
                    if is_unique:
                        hist, edges = np.array([1]), vec[0] + np.array([-0.5, 0.5])
                    else:
                        hist, edges = hist_data(vec, limits, nbins=nbins)
                    yield {'par' : par,
                           'name' : name,
                           'vec' : vec,
                           'hist' : (hist, edges),
                           'unique' : is_unique}
                else:
                    if is_unique:
                        x_kde, y_kde = np.array([vec[0], vec[0]]), np.array([0,1])
                    else:
                        x_kde, y_kde = kde_data(vec, limits, c=c_kde)
                    yield {'par': par,
                           'name' : name,
                           'vec' : vec,
                           'kde' : (x_kde, y_kde),
                           'unique' : is_unique}
            else:
                yield {'par' : par,
                       'name' : name,
                       'vec' : vec,
                       'unique' : is_unique}

def forestplot_data(fit, pars, dtypes=None, split_pars=False, **kwargs):
    """Function to yield data for the forestplot.
    Histogram and kde height data is normalized to 1

    Parameters
    ----------
    fit : StanFit4Model object
    pars : tuple
        Parameters plotted.
    dtypes : dict, optional
        Dictionary containing parameters as a key value and transform-function value.
    split_pars : bool, optional, default False
        If True plot each parameter including vector and matrix components in their own axis.
    c_kde : int, optional
        Constant for the kde function, ignored if density==False.
    nbins : int, optional
        Maximum number of bins for histogram function, ignored if density==False.
    inc_warmup : bool, optional, default False
        Include warmup.
    overlap : float, optional, default 0.98
        Overlap density plots.

    Yields
    ------
    data : dict
        dict contains
            par, parameter name
            name, parameter component name
            vec,
            hist, histogram data, optional
            kde, kde data, optional
    """
    # TODO: Add option to plot chains independently
    # This can be done with np.ndindex tricks
    c_kde = kwargs.get('c_kde', 1)
    nbins = kwargs.get('nbins', None)
    inc_warmup = kwargs.get('inc_warmup', False)
    sampler_params = {'accept_stat__', 'stepsize__', 'treedepth__', \
                      'n_leapfrog__', 'divergent__', 'energy__'}
    overlap = kwargs.get('overlap', 0.98)
    for par in pars:
        if par in sampler_params:
            n_save = fit.sim['n_save'][0]
            if inc_warmup:
                n_save = n_save - fit.sim['warmup']
            vecs = np.concatenate([sp['par'][-n_save:] for sp in fit.get_sampler_params()])
        else:
            vecs = fit.extract(pars=par, permuted=False, dtypes=dtypes, inc_warmup=inc_warmup)[par]
        nchains = fit.sim['chains']
        if nchains == 1:
            if len(vecs.shape) == 1:
                # add chain dimension
                vecs = np.expand_dims(vecs, -1)
        par_dims = vecs.shape[2:]
        if not par_dims:
            vecs = np.expand_dims(vecs, -1)
            par_dims = vecs.shape[2:]
        # reshape and translate for plotting
        vecs = vecs.reshape([vecs.shape[0]*vecs.shape[1]]+list(par_dims), order='F').T
        if split_pars:
            limits = None
        else:
            limits = vecs.min(), vecs.max()
        m = np.multiply.reduce(par_dims)
        indices = np.c_[np.unravel_index(np.arange(m), par_dims, order='F')]
        for idx in indices:
            vec = vecs[tuple(idx)]
            is_unique = len(np.unique(vec)) == 1
            # use par if value scalar
            if par_dims == (1,):
                name = par
            else:
                # add 1 for the index
                name = "{}[{}]".format(par, ",".join(map(str, idx+1)))
            if isinstance(vec[0], (int, np.integer, np.uint)):
                if is_unique:
                    hist, edges = np.array([1]), vec[0] + np.array([-0.5, 0.5])
                else:
                    hist, edges = hist_data(vec, limits, nbins=nbins)
                yield {'par' : par,
                       'name' : name,
                       'vec' : vec,
                       'hist' : (hist/hist.max()*overlap, edges),
                       'unique' : is_unique}
            else:
                if is_unique:
                    x_kde, y_kde = np.array([vec[0], vec[0]]), np.array([0,1])
                else:
                    x_kde, y_kde = kde_data(vec, limits, c=c_kde)
                yield {'par': par,
                       'name' : name,
                       'vec' : vec,
                       'kde' : (x_kde, y_kde/y_kde.max()*overlap),
                       'unique' : is_unique}

def mcmc_parcoord_data(fit, pars, divergence=False, **kwargs):
    """Function to gather data for the mcmc_parcoord-plot.

    Parameters
    ----------
    fit : StanFit4Model object
    pars : tuple
        Parameters plotted.
    divergence : bool, optional
        If True, return divergent sample independently.
    transform : str, function
        If str, use {'minmax', 'standard'}
            minmax = (arr - min) / (max - min)
            standard = (arr - mean) / std
        If function, transform data with tranform-function.
        E.g. function=np.log, arr = np.log(arr)
    inc_warmup : bool, optional

    Returns
    -------
    data : list
        List contains = (names, data, divergent data (if divergence is True)).
    """
    inc_warmup = kwargs.pop('inc_warmup', False)
    sampler_params = {'accept_stat__', 'stepsize__', 'treedepth__', \
                      'n_leapfrog__', 'divergent__', 'energy__'}
    vectors = []
    names = []
    transform = kwargs.pop('transform', None)
    for par in pars:
        if par in sampler_params:
            n_save = fit.sim['n_save'][0]
            if inc_warmup:
                n_save = n_save - fit.sim['warmup']
            vecs = np.concatenate([sp['par'][-n_save:] for sp in fit.get_sampler_params()])
        else:
            vecs = fit.extract(pars=par, permuted=False, inc_warmup=inc_warmup)[par]
        par_dims = vecs.shape[2:]
        if not par_dims:
            vecs = np.expand_dims(vecs, -1)
            par_dims = vecs.shape[2:]
        m = np.multiply.reduce(par_dims)
        vecs = vecs.reshape([vecs.shape[0]*vecs.shape[1]]+list(par_dims), order='F')
        vecs = vecs.reshape(-1, m, order='F')
        if transform is not None:
            if isinstance(transform, str):
                if transform.lower() in ('min-max', 'minmax'):
                    vmin, vmax = vecs.min(0), vecs.max(0)
                    vecs = (vecs - vmin) / (vmax-vmin)
                elif transform.lower() in ('standard'):
                    vmean, vstd = vecs.mean(0), vecs.std(0, ddof=1)
                    vecs = (vecs - vmean) / vstd
        indices = np.c_[np.unravel_index(np.arange(m), par_dims, order='F')]
        if m > 1:
            names.extend(["{}[{}]".format(par, ",".join(map(str, idx+1))) for idx in indices])
        else:
            names.append(par)
        vectors.append(vecs)
    vectors = np.column_stack(vectors)
    if not divergence:
        return names, vectors.T, []
    else:
        n_save = fit.sim['n_save'][0]
        if not inc_warmup:
            n_save = n_save - fit.sim['warmup2'][0]
        div_vec = np.r_[[item['divergent__'][-n_save:] for item in fit.get_sampler_params()]].ravel().astype(bool)
        if div_vec.sum():
            return names, vectors[~div_vec, :].T, vectors[div_vec, :].T
        else:
            return names, vectors.T, []
