import pandas as pd
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import re


def header_decomposition(header: List[str]) -> List[dict]:
    '''
    Decompose the header and return the List of the dictionaries.
    The decomposition is::
        string<key1=value1><key2=value2>
        -> {'label': 'string', 'key1': 'value1', 'key2': 'value2'}
    
    The key of the string not in <> is 'label'.
    Meanwhile the string inside <>, the part before '=' is the key,
    and the part after '=' is the value.
    
    Parameters
    ----------
    header : List[str]
        Header to decompose.
    
    Returns
    -------
    info : List[dict]
        List of the decomposed header.
    Examples
    --------
    >>> h = [AAA<V=W><X=Y>,BBB<X=Z>]
    >>> info = header_decomposition(h)
    >>> info
    [{'label': 'AAA', 'V': 'W', 'X': 'Y'}, {'label': 'BBB', 'X': 'Z'}]
    '''
    info : List[dict] = []
    for name in header:
        d = dict()
        label = re.findall('\A[^<]+', name)[0]
        d['label'] = label
        params = re.findall('<[a-z]+=[a-z]+>', name)
        for param in params:
            p = re.findall('[a-z]+', param)
            d[p[0]] = p[1]

        info.append(d)

    return info


def performance_profile(path: str, stop: float=5., step: float=1e-2, tau: str=None, grid: bool=True) -> None:
    '''
    Plot the performance profile.
    Parameters
    ----------
    path : str
        Input file to plot.
    
    stop : float = 5.0
        Max value of the x-axis.
    step : float = 1e-2
        Spacing between values.
    
    tau : str = None
        Details of tau (e.g., elapsed time).
        If None, the x-axis label is just 'tau'.
    
    grid : bool = True
        Whether to show the grid lines.
    '''
    df : pd.DataFrame = pd.read_csv(path)

    df = df.drop(df.columns[[3, 4, 5, 9, 10, 11]], axis=1)
    # df = df.drop(df.columns[[0, 1, 2, 6, 7, 8]], axis=1)

    header : List[str] = df.columns.values.tolist()
    data : np.ndarray = df.values

    if np.min(data) <= 0.:
        raise ValueError('All values ​​in the data must be positive.')

    num_p : int = data.shape[0]
    num_s : int = data.shape[1]
    r : np.ndarray = data.T / np.min(data, axis=1)

    info : List[dict] = header_decomposition(header)

    def _pp(t: float, index: int) -> float:
        return np.count_nonzero(r[index] <= t) / num_p

    pp = []
    for idx in range(num_s):
        _temp = []
        x = np.arange(1, stop, step)
        for val in x:
            _temp.append(_pp(val, idx))
        pp.append(_temp)

    _pp_plot(pp, info, stop, step, tau, grid)


def _pp_plot(pp, info, stop, step, tau, grid) -> None:
    x : np.ndarray = np.arange(1, stop, step)
    for idx, y in enumerate(pp):
        plt.plot(x, y, **info[idx])

    plt.xlim(1, stop)
    if tau is None:
        plt.xlabel(r'$\tau$')
    else:
        plt.xlabel(r'$\tau$' + f' ({tau})')
    plt.ylabel(r'$P_s(\tau)$')
    plt.legend(loc='lower right')
    if grid:
        plt.grid()

    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.1, top=1)
    plt.show()


if __name__ == '__main__':
    # performance_profile('results/rayleigh/iterations.csv')
    performance_profile('results/off-diag/iterations.csv')