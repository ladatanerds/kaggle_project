import json
import warnings
warnings.filterwarnings("ignore")

from random import sample
from itertools import product
import itertools


def random_search(num_searches=40, **kwargs):
    """
    This implements random search for hyper-parameter search.

    Parameters
    ----------
    num_searches : int
        The number of hyper-parameter searches.
    kwargs : dict
        A dictionary, where the key is the hyper-parameter, and the value is a list of possible hyper-parameter values

    Returns
    -------
    list
        A list of dicts in which each dict is a set of hyper-parameter choices for all the hyper-parameters

    Notes
    ------
    Lot of content taken from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists.
    """
    # hyper-parameter arg names
    keys = kwargs.keys()
    # hyper-parameter options
    vals = kwargs.values()
    product_dts = []
    product_dts_json = []

    for _ in range(num_searches):
        not_contained = False
        # check to make sure hyper-parameter options haven't previously been seen
        while not not_contained:
            # retrieve the arg-values tuples for all hyper-parameter args in random order
            keys_and_vals = sample(list(zip(keys, vals)), len(keys))
            keys, vals = zip(*keys_and_vals)
            # randomize the hyper-parameter options for each hyper-parameter arg
            vals = [sample(val, len(val)) for val in vals]
            # get one set of possible hyper-parameters for a model
            instance = next(product(*vals))
            # convert the instance to a dictionary
            product_dt = dict(zip(keys, instance))
            # get the json string from the dict
            product_dt_json = json.dumps(product_dt, sort_keys=True)
            # add the instance to the experiment list if json hasn't been seen
            if product_dt_json not in product_dts_json:
                product_dts.append(product_dt)
                product_dts_json.append(product_dt_json)
                not_contained = True
    return product_dts


# grid search function
def grid_search(**kwargs):
    """
    This implements grid search for hyper-parameter search.

    Parameters
    ----------
    kwargs : dict
        A dictionary, where the key is the hyper-parameter, and the value is a list of possible hyper-parameter values

    Returns
    -------
    list
        A list of dicts in which each dict is a set of hyper-parameter choices for all the hyper-parameters

    Notes
    ------
    Lot of content taken from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists.
    """
    # hyper-parameter arg names
    keys = kwargs.keys()
    # hyper-parameter options
    vals = kwargs.values()
    searches = []
    # get cartesian product from all hyper-parameter options
    for instance in itertools.product(*vals):
        searches.append(dict(zip(keys, instance)))
    return searches