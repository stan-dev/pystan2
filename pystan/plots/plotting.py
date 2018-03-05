import logging
logger = logging.getLogger('pystan')

import numpy as np
from pystan._compat import string_types
from copy import deepcopy

from .plotting_data import hist_data, kde_data, traceplot_data, forestplot_data, mcmc_parcoord_data

def plot_hist(hist, edges, ax, fill, fill_dict, zero_level=None, **kwargs):
    """Function to plot histogram

    Parameters
    ----------
    hist : ndarray
    edges : ndarray
    ax : Axes instance
        Axis to plot on.
    fill : bool 
        Fill the histogram with color.
    fill_dict : dict 
        If fill==True, append to `fill_between` function.
    zero_level: float, optional
        Value used as a zero level, default is 0.
    kwargs:
        Dictionary is appended to `plot` function.

    Returns
    -------
    ax : Axes instance
    """
    if zero_level is None:
        zero_level = 0
    #remove unique keyword
    unique = kwargs.pop('unique', False)
    x_hist = edges-(edges[1]-edges[0])/2
    x_hist = np.r_[x_hist[0], x_hist, x_hist[-1]]
    y_hist = np.r_[zero_level,hist[0],hist,zero_level]

    default_dict = {
        ('lw', 'linewidth') : 2,
        ('c', 'color') : None,
        'zorder' : 30,
    }
    # set default values
    for key, value in default_dict.items():
        if isinstance(key, tuple):
            if any(key_ in kwargs for key_ in key):
                continue
            if value is not None:
                kwargs[key[0]] = value
        else:
            if key in kwargs:
                continue
            if value is not None:
                kwargs[key] = value

    hist_plot, = ax.step(x_hist, y_hist, **kwargs)
    if fill:
        fill_dict = deepcopy(fill_dict)

        default_dict = {
            'color' : hist_plot.get_color(),
            'alpha' : 0.2,
            'zorder' : kwargs.get('zorder', 30),
        }

        # set default values
        for key, value in default_dict.items():
            if isinstance(key, tuple):
                if any(key_ in fill_dict for key_ in key):
                    continue
                if value is not None:
                    fill_dict[key[0]] = value
            else:
                if key in fill_dict:
                    continue
                if value is not None:
                    fill_dict[key] = value

        # plot face
        ax.fill_between(x_hist, y_hist, zero_level, step="pre", **fill_dict)
    return ax

def plot_kde(x, y, ax, fill, fill_dict, zero_level=None, **kwargs):
    """Function to plot kernel density estimation

    Parameters
    ----------
    x : ndarray
    y : ndarray
    ax : Axes instance
        Axis to plot on.
    fill : bool
        Fill the kde with color. 
    fill_dict : dict
        If fill == True, append to `fill_between` function.
    zero_level: float, optional
        Value used as a zero level, default is 0.
    kwargs:
        Instance is appended to `plot` function

    Returns
    -------
    ax : Axes instance
    """
    if zero_level is None:
        zero_level = 0
    unique = kwargs.pop('unique', False)
    default_dict = {
        ('lw', 'linewidth') : 2,
        ('c', 'color') : None,
        'zorder' : 30,
    }
    # set default values
    for key, value in default_dict.items():
        if isinstance(key, tuple):
            if any(key_ in kwargs for key_ in key):
                continue
            if value is not None:
                kwargs[key[0]] = value
        else:
            if key in kwargs:
                continue
            if value is not None:
                kwargs[key] = value

    # plot edge
    kde_plot, = ax.plot(x, y, **kwargs)
    if fill and not unique:
        fill_dict = deepcopy(fill_dict)

        default_dict = {
            'color' : kde_plot.get_color(),
            'alpha' : 0.2,
            'zorder' : kwargs.get('zorder', 30),
        }

        # set default values
        for key, value in default_dict.items():
            if isinstance(key, tuple):
                if any(key_ in fill_dict for key_ in key):
                    continue
                if value is not None:
                    fill_dict[key[0]] = value
            else:
                if key in fill_dict:
                    continue
                if value is not None:
                    fill_dict[key] = value

        # plot face
        ax.fill_between(x, y, zero_level, **fill_dict)
    return ax

def _plot_statistic_for_density(density_x, density_y, vec, ax, plot_dict, zero_level=0, method=None):
    """Function to plot kernel density estimation

    Parameters
    ----------
    density_x : ndarray
        x-values for density data.
    density_y : ndarray
        y-values for density data.
    vec : ndarray
        Original data.
    ax : Axes instance
        Axis to plot on.
    plot_dict : dictionary
        dictionary appended to `plot` -function.
        Needs to have a key 'func' containing function for the statistic.
    zero_level: float, optional
        Value used as a zero level, default is 0.
    method:
        If method == 'hist' or 'histogram' use correct height for the statistic.
        
    Returns
    -------
    ax : Axes instance
    """
    plot_dict = plot_dict.copy()
    statistic_func = plot_dict.pop('func')
    for key, item in {'c' : 'k', 'lw' : 2}.items():
        if key not in plot_dict:
            plot_dict[key] = item
    statistic = statistic_func(vec)
    # find height
    loc_x_max = (density_x > statistic).argmax()
    loc_x_min = loc_x_max - 1
    x_max = density_x[loc_x_max]
    x_min = density_x[loc_x_min]
    y_max = density_y[loc_x_max]
    y_min = density_y[loc_x_min]
    # weighted average
    w = (x_max - statistic) / (x_max - x_min)
    height = w * y_min + (1-w) * y_max
    if method in ('hist', 'histogram'):
        height = y_min
    # plot statistic
    ax.plot([statistic, statistic],[zero_level, height], **plot_dict)
    return ax


def traceplot(fit, pars=None, dtypes=None, kde_dict=None, hist_dict=None, **kwargs):
    """
    Use traceplot to display parameters.

    Parameters
    ----------
    fit : StanFit4Model object
    pars : tuple, optional
        Parameters used for the plotting.
    dtypes : dict, optional
        Dictionary containing parameters as keys and transform-functions as values.
    kde_dict: dictionary, optional
        Dictionary appended for kde plots as kwargs.
    hist_dict: dictionary, optional
        Dictionary appended for histogram plots as kwargs.
    split_pars : bool, optional
        If True plot each parameter including vector and matrix components in their own axis.
    density : bool, optional
        If False don't plot kde or histograms. Plots histogram if dtype for parameter is int.
    statistic : bool or str or list of dictionaries, optional
        If True, statistic = mean.
        If str, {'mean', 'median'}
        Add statistic line for the density plot. Statistic function (key='func') is popped out.
        The rest of the dictionary is appended to plt.plot.
        E.g. [{'func' : np.mean, 'lw' : 1}].
        Use functools.partial if statistic funtion needs parameters.
    force : bool, optional
        If force is True then plot large number of parameters.
    figsize : tuple, optional
        Given in inches.
    fill : bool, optional
        Fill the density plot.
    fill_dict : dict, optional
        Keywords appended to `fill_between` plot.
    tight_layout : bool, optional
        Padding is set to 0.5.
    cmap : str or colormap object
        Color lines based on the colormap.
        Uses 0.5 for scalars
        else `np.linspace(0,1,n)` where n is the vector count

    Returns
    -------
    fig : Figure instance
    axes : ndarray of Axes instances
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.critical("matplotlib required for plotting.")
        raise
    if pars is None:
        pars = fit.model_pars
    elif isinstance(pars, string_types):
        pars = [pars]

    if kde_dict is None:
        kde_dict = {}

    if hist_dict is None:
        hist_dict = {}
    
    # number of rows
    split_pars = kwargs.pop('split_pars', False)
    if not split_pars:
        n = len(pars)
    else:
        # skip lp__
        model_pars = fit.model_pars
        n = 0
        for parameter in pars:
            par_index = model_pars.index(parameter)
            par_dims = fit.par_dims[par_index]
            if len(par_dims) == 0:
                par_dims = [1]
            n += np.multiply.reduce(par_dims)
    # number of columns
    # define if density is plotted
    density = kwargs.pop('density', True)
    density_col = 0
    if density:
        trace_col = 1
        m = 2
    else:
        trace_col = 0
        m = 1
    # raise an exception if too many subplots
    force = kwargs.pop('force', False)
    if not force:
        if n > 20:
            err_msg = "Too many subplots: (row={},col={})->total={},"\
                      " please use 'force' keyword to force"\
                      " enable plotting".format(n, m, n*m)
            logger.critical(err_msg)
            raise ValueError(err_msg)
    # create figure object
    figsize = kwargs.pop('figsize', (6, max(4.5, 1.5*n)))
    fig, axes = plt.subplots(n, m, figsize=figsize, squeeze=False)
    legend = kwargs.pop('legend', False)
    fill = kwargs.pop('fill', True)
    fill_dict = kwargs.pop('fill_dict', dict())
    
    tight_layout = kwargs.pop('tight_layout', True)
    
    cmap = kwargs.get('cmap', None)
    cmap_dict = dict()
    if cmap is not None and isinstance(cmap, string_types):
        cmap = plt.cm.get_cmap(cmap)    
        if not split_pars:
            for parameter in pars:
                par_index = fit.model_pars.index(parameter)
                par_dims = fit.par_dims[par_index]
                if len(par_dims) == 0:
                    cmap_dict[parameter] = np.array([cmap(0.5)])
                else:
                    cmap_dict[parameter] = cmap(np.linspace(0,1,np.multiply.reduce(par_dims)))
        else:
            for parameter in pars:
                cmap_dict[parameter] = np.array([cmap(0.5)])
    
    statistic = kwargs.pop('statistic', False)
    if isinstance(statistic, int) and statistic:
        statistic = [{'func' : np.mean}]
    elif isinstance(statistic, str) and statistic:
        if statistic == 'mean':
            statistic = [{'func' : np.mean}]
        elif statistic == 'median':
            statistic = [{'func' : np.median}]
    statistic_color = kwargs.pop('statistic_color', 'k')
    if statistic:
        for i, statistic_dict in enumerate(statistic):
            if 'color' not in statistic_dict:
                statistic[i]['color'] = statistic_color
    
    # use this if split_pars is False, else use i
    ax_row = 0
    last_parameter = ''
    for i, traceplot_dict in enumerate(traceplot_data(fit, pars, dtypes, **kwargs)):
        parameter = traceplot_dict['par']
        if split_pars:
            ax_row = i
        else:
            # update row
            if i == 0:
                if parameter in cmap_dict:
                    axes[ax_row, density_col].set_prop_cycle(color=cmap_dict[parameter])
                    axes[ax_row, trace_col].set_prop_cycle(color=cmap_dict[parameter])
                last_parameter = parameter
            else:
                if parameter != last_parameter:
                    # update last axis and move to next row
                    axes[ax_row, density_col].set_title(last_parameter)
                    axes[ax_row, trace_col].set_title(last_parameter)
                    axes[ax_row, density_col].set_ylabel('density')
                    axes[ax_row, trace_col].set_xlabel('index')
                    if legend:
                        axes[ax_row, density_col].set_legend()
                    ax_row += 1
                    if parameter in cmap_dict:
                        axes[ax_row, density_col].set_prop_cycle(color=cmap_dict[parameter])
                        axes[ax_row, trace_col].set_prop_cycle(color=cmap_dict[parameter])
                    last_parameter = parameter
        
        if 'kde' in traceplot_dict:
            plot_kde(*traceplot_dict['kde'], ax=axes[ax_row, density_col], fill=fill, fill_dict=fill_dict, **kde_dict)
            if statistic:
                for statistic_dict in statistic:
                    _plot_statistic_for_density(traceplot_dict['kde'][0],\
                                                traceplot_dict['kde'][1],\
                                                traceplot_dict['vec'],\
                                                axes[ax_row, density_col],\
                                                plot_dict=statistic_dict, zero_level=0)
        elif 'hist' in traceplot_dict:
            plot_hist(*traceplot_dict['hist'], ax=axes[ax_row, density_col], fill=fill, fill_dict=fill_dict, **hist_dict)
            if statistic:
                for statistic_dict in statistic:
                    edges = traceplot_dict['hist'][1]
                    hist = traceplot_dict['hist'][0]
                    x_hist = edges-(edges[1]-edges[0])/2
                    y_hist = hist
                    _plot_statistic_for_density(x_hist, y_hist,\
                                                traceplot_dict['vec'],
                                                axes[ax_row, density_col],\
                                                plot_dict=statistic_dict, zero_level=0,\
                                                method='hist')
        # plot trace
        vec = traceplot_dict['vec']
        axes[ax_row,trace_col].plot(vec)
    # update axis labels
    axes[ax_row, density_col].set_title(parameter)
    axes[ax_row, trace_col].set_title(parameter)
    axes[ax_row, density_col].set_ylabel('density')
    axes[ax_row, trace_col].set_xlabel('trace')
    if legend:
        axes[ax_row, density_col].legend()
    if tight_layout:
        fig.tight_layout(pad=0.5)
    return fig, axes

def forestplot(fit, pars=None, dtypes=None, kde_dict=None, hist_dict=None, **kwargs):
    """
    Use forestplot to display parameters.

    Parameters
    ----------
    fit : StanFit4Model object
    pars : tuple, optional
        Parameters for the plotting
    dtypes : dict, optional
        Dictionary containing parameters as a key value and transform-functions as values.
    kde_dict: dictionary, optional
        Dictionary appended for kde plots as kwargs.
    hist_dict: dictionary, optional
        Dictionary appended for histogram plots as kwargs.
    statistic : bool or str or list of dictionaries, optional
        If True, statistic = mean.
        If str, {'mean', 'median'}
        Add statistic line for the density plot. Statistic function (key='func') is popped out.
        The rest of the dictionary is appended to plt.plot.
        E.g. [{'func' : np.mean, 'lw' : 1}].
        Use functools.partial if statistic funtion needs parameters.
    force : bool, optional
        If force is True then plot large number of parameters.
    legend : bool, optional
        Add legend for the plot.
    figsize : tuple, optional
        Given in inches.
    fill : bool, optional
        Fill the density plot.
    fill_dict : dict, optional
        Keywords appended to `fill_between` plot.
    tight_layout : bool, optional
        Padding is set to 0.5.

    Returns
    -------
    fig : Figure instance
    ax : Axes instances
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.critical("matplotlib required for plotting.")
        raise
    if pars is None:
        pars = fit.model_pars
    elif isinstance(pars, string_types):
        pars = [pars]

    if kde_dict is None:
        kde_dict = {}

    if hist_dict is None:
        hist_dict = {}

    # number of rows
    # skip lp__
    model_pars = fit.model_pars
    n = 0
    for parameter in pars:
        par_index = model_pars.index(parameter)
        par_dims = fit.par_dims[par_index]
        if len(par_dims) == 0:
            par_dims = [1]
        n += np.multiply.reduce(par_dims)
    
    # create figure object
    figsize = kwargs.pop('figsize', (6, max(4.5, n)))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    legend = kwargs.pop('legend', False)
    fill = kwargs.pop('fill', True)
    fill_dict = kwargs.pop('fill_dict', dict())
    tight_layout = kwargs.pop('tight_layout', True)
    
    cmap = kwargs.get('cmap', None)
    if cmap is not None:
        if isinstance(cmap, string_types):
            if cmap not in plt.cm.cmap_d.keys():
                logger.warning("cmap was not recognized. Using viridis")
                cmap = 'viridis'
            cmap = plt.cm.get_cmap(cmap)
        if n == 1:
            ax.set_prop_cycle(color=[cmap(0.5)])
        else:
            ax.set_prop_cycle(color=cmap(np.linspace(0,1,n)))
    
    statistic = kwargs.pop('statistic', False)
    if isinstance(statistic, int) and statistic:
        statistic = [{'func' : np.mean}]
    elif isinstance(statistic, str) and statistic:
        if statistic == 'mean':
            statistic = [{'func' : np.mean}]
        elif statistic == 'median':
            statistic = [{'func' : np.median}]
    statistic_color = kwargs.pop('statistic_color', 'k')
    if statistic:
        for i, statistic_dict in enumerate(statistic):
            if 'color' not in statistic_dict:
                statistic[i]['color'] = statistic_color
    last_parameter = ''
    name_list = []
    height_list = []
    for row, forestplot_dict in enumerate(forestplot_data(fit, pars, dtypes, **kwargs)):
        parameter = forestplot_dict['par']

        # update row
        if row == 0:
            last_parameter = parameter
        else:
            last_parameter = parameter
         
            
        if 'hist' in forestplot_dict:
            
            hist, edges = forestplot_dict['hist']
            hist = hist - row

            # zorder
            if 'zorder' not in hist_dict:
                zorder = 30+row
            else:
                zorder = hist_dict.pop('zorder')
            plot_hist(hist, edges, ax=ax, fill=fill, fill_dict=fill_dict, zero_level=-row, zorder=zorder, **hist_dict)
            if statistic:
                x_hist = edges-(edges[1]-edges[0])/2
                y_hist = hist
                for statistic_dict in statistic:
                    statistic_dict = statistic_dict.copy()
                    if 'zorder' not in statistic_dict:
                        statistic_dict['zorder'] = zorder
                    _plot_statistic_for_density(x_hist, y_hist,\
                                                forestplot_dict['vec'],\
                                                ax,\
                                                plot_dict=statistic_dict, zero_level=-row,\
                                                method='hist')
        else:
                                  
            x_kde, y_kde = forestplot_dict['kde']
            y_kde = y_kde - row
            # zorder
            if 'zorder' not in kde_dict:
                zorder = 30+row
            else:
                zorder = kde_dict.pop('zorder')
            plot_kde(x_kde, y_kde, ax=ax, fill=fill, fill_dict=fill_dict, zero_level=-row, zorder=zorder, **kde_dict)
            if statistic:
                for statistic_dict in statistic:
                    statistic_dict = statistic_dict.copy()
                    if 'zorder' not in statistic_dict:
                        statistic_dict['zorder'] = zorder
                    _plot_statistic_for_density(x_kde, y_kde,\
                                                forestplot_dict['vec'],\
                                                ax,\
                                                plot_dict=statistic_dict, zero_level=-row)

        name_list.append(forestplot_dict['name'])
        height_list.append(-row)

    # update axis labels
    ax.set_yticks(height_list)
    ax.set_yticklabels(name_list)

    if legend:
        ax.legend()
    if tight_layout:
        fig.tight_layout(pad=0.5)
    return fig, ax

def mcmc_parcoord(fit, pars=None, transform=None, divergence=None, **kwargs):
    """
    mcmc_parcoord-plot enable simultaneous examinations of multiple dimensions.

    Parameters
    ----------
    fit : StanFit4Model object
    pars : tuple, optional
        Parameters for the plotting
    transform : str or function, optional
        If str, {'min'
        Function to transform data.
        Must return data in original shape.
    divergence : bool or str or cmap, optional
        If True or non-empty dictionary plot divergent samples independently.
        If divergence is a color, color divergent samples correspondingly.
        If divergence is a cmap (str, function), color divergent samples correspondingly.
    cmap : str or function, optional
        Color samples based on the index (order) with chosen colormap
    color : str, tuple, optional
        Color samples based on the one color.
    alpha : float, optional
        Alpha value used for the data
    lw : float or int, optional
        Linewidth for the data.
    alpha_div : float, optional
        Alpha value used for the divergent data
    lw_div : float or int, optional
        Linewidth for the divergent data.
    label : str, optional
        Add label for the plot
    figsize : tuple, optional
        Given in inches.
    legend : bool, optional
        Add legend for the plot.
    tight_layout : bool, optional
        Padding is set to 0.5.

    Returns
    -------
    fig : Figure instance
    ax : Axes instances
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.critical("matplotlib required for plotting.")
        raise
    if pars is None:
        pars = fit.model_pars
    elif isinstance(pars, string_types):
        pars = [pars]

    # create figure
    figsize = kwargs.pop('figsize', (6, 4))
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    legend = kwargs.pop('legend', True if bool(divergence) else False)

    tight_layout = kwargs.pop('tight_layout', True)

    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', kwargs.pop('c', None))

    alpha = kwargs.pop('alpha', 0.5)
    lw = kwargs.pop('lw', kwargs.get('linewidth', 1))

    alpha_div = kwargs.pop('alpha_div', 1)
    lw_div = kwargs.pop('lw_div', lw)

    label = kwargs.pop('label', 'Non-divergent')

    names, data, data_div = mcmc_parcoord_data(fit, pars=pars, \
                            transform=transform, divergence=bool(divergence))
    # run parallel coordinates plot in one-step
    if cmap is None:
        if color is None:
            color = 'k'
        ax.plot(data, c=color, alpha=alpha, lw=lw, **kwargs)
    # run line by line
    else:
        n = data.shape[1]
        if isinstance(cmap, str):
            colors = plt.cm.get_cmap(cmap)(np.linspace(0,1,n))
        else:
            colors = cmap(np.linspace(0,1,n))
        # non-divergent samples
        for i, _data in enumerate(data.T):
            ax.plot(_data, c=colors[i], alpha=alpha, lw=lw, **kwargs)
    if divergence and len(data_div):
        n_div = data_div.shape[1]
        # set colors / cmap
        if isinstance(divergence, bool):
            divergence = ['lime']*n_div
        elif isinstance(divergence, str) and divergence in plt.cm.cmap_d.keys():
            divergence = plt.cm.get_cmap(divergence)(np.linspace(0,1,n_div))
        elif not isinstance(divergence, str) and divergence:
            divergence = divergence(np.linspace(0,1,n_div))
        elif isinstance(divergence, str):
            divergence = [divergence]*n_div
        for i, _data in enumerate(data_div.T):
            if i == 0:
                label = 'Divergent'
            else:
                label = '_no_legend_'
            ax.plot(_data, c=divergence[i], alpha=alpha_div, lw=lw_div, label=label, **kwargs)

    if legend:
        ax.legend()
    if tight_layout:
        fig.tight_layout(pad=0.5)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90)

    return fig, ax
