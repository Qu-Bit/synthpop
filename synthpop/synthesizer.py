#synthesizer
#semcog multiprocessing version
import logging
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
import math
import multiprocessing as mp

from scipy.stats import chisquare
from . import categorizer as cat
from . import draw
from .ipf.ipf import calculate_constraints
from .ipu.ipu import household_weights
import time
import psutil
import os
import gc



logger = logging.getLogger("synthpop")
FitQuality = namedtuple(
    'FitQuality',
    ('people_chisq', 'people_p'))
BlockGroupID = namedtuple(
    'BlockGroupID', ('state', 'county', 'tract', 'block_group'))


def enable_logging():
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def rebalance(x0, tar, w=None):
    #adjust household size distribution based on population targets
    #x0: # of hhs by hh size (list/vector)
    #tar: target population by geo unit(block group, tract, etc)
    #weight: for HHsize 7+, the average size in this block group (based on first synthesis run)
    x = x0.copy()
    hh = x.sum()
    rx = np.array(range(len(x0)))
    if w is None:
        w = rx + 1
    while x.dot(w) < tar and x.max() < hh:
        # print tar, x.dot(w), x
        i = np.random.choice(rx[:-1], p=(1.0 * x[:-1] / x[:-1].sum()))
        j = np.random.choice(rx[i:], p=(1.0 * x[i:] / x[i:].sum()))
        x[i] -= 1
        x[j] += 1
    return x


def synthesize(h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
               marginal_zero_sub=.01, jd_zero_sub=.001, hh_index_start=0):

    # this is the zero marginal problem
    h_marg = h_marg.replace(0, marginal_zero_sub)
    p_marg = p_marg.replace(0, marginal_zero_sub)

    # zero cell problem
    h_jd.frequency = h_jd.frequency.replace(0, jd_zero_sub)
    p_jd.frequency = p_jd.frequency.replace(0, jd_zero_sub)

    # ipf for households
    logger.info("Running ipf for households")
    h_constraint, _ = calculate_constraints(h_marg, h_jd.frequency)
    h_constraint.index = h_jd.cat_id

    logger.debug("Household constraint")
    logger.debug(h_constraint)

    # ipf for persons
    logger.info("Running ipf for persons")
    p_constraint, _ = calculate_constraints(p_marg, p_jd.frequency)
    p_constraint.index = p_jd.cat_id

    logger.debug("Person constraint")
    logger.debug(p_constraint)

    # make frequency tables that the ipu expects
    household_freq, person_freq = cat.frequency_tables(p_pums, h_pums,
                                                       p_jd.cat_id,
                                                       h_jd.cat_id)

    # do the ipu to match person marginals
    logger.info("Running ipu")
    import time
    t1 = time.time()
    best_weights, fit_quality, iterations = household_weights(household_freq,
                                                              person_freq,
                                                              h_constraint,
                                                              p_constraint)
    logger.info("Time to run ipu: %.3fs" % (time.time()-t1))

    logger.debug("IPU weights:")
    logger.debug(best_weights.describe())
    logger.debug("Fit quality:")
    logger.debug(fit_quality)
    logger.debug("Number of iterations:")
    logger.debug(iterations)

    num_households = int(h_marg.groupby(level=0).sum().mean())
    #print "Drawing %d households" % num_households

    best_chisq = np.inf

    return draw.draw_households(
        num_households, h_pums, p_pums, household_freq, h_constraint,
        p_constraint, best_weights, hh_index_start=hh_index_start)


def synthesize_one(recipe, geog_id,
                   marginal_zero_sub=.01, jd_zero_sub=.001):

    #print "Synthesizing [state, county, tract, block group]:", geog_id.values
    sys.stdout.flush()
    
    h_marg = recipe.get_household_marginal_for_geography(geog_id)
    logger.debug("Household marginal")
    logger.debug(h_marg)

    p_marg = recipe.get_person_marginal_for_geography(geog_id)
    logger.debug("Person marginal")
    logger.debug(p_marg)

    h_pums, h_jd = recipe.\
        get_household_joint_dist_for_geography(geog_id)
    logger.debug("Household joint distribution")
    logger.debug(h_jd)

    p_pums, p_jd = recipe.get_person_joint_dist_for_geography(geog_id)
    logger.debug("Person joint distribution")
    logger.debug(p_jd)

    if (hasattr(recipe, "hh_size_order") and
        hasattr(recipe, "get_hh_size_weight") and
        hasattr(recipe, "get_hh_size_person_factor")):
        orig_hh_size = h_marg.loc["hh_size"].loc[recipe.hh_size_order]

        pmax=max(p_marg.groupby(level=0).sum())
        if math.isnan(pmax):
            pmax=0
        h_marg.loc["hh_size"].loc[recipe.hh_size_order] = rebalance(
            orig_hh_size,
            tar=pmax*float(recipe.get_hh_size_person_factor(geog_id)),
            w=np.array(recipe.get_hh_size_weight(geog_id)))

    households, people, people_chisq, people_p = \
        synthesize(
            h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
            marginal_zero_sub=marginal_zero_sub, jd_zero_sub=jd_zero_sub,
            hh_index_start=0)    

    #covert maringal to dataframe
    h_marg=h_marg.to_frame(name='marginal')
    h_marg.reset_index(inplace=True)
    p_marg=p_marg.to_frame(name='marginal')
    p_marg.reset_index(inplace=True)

    # Append location identifiers to the synthesized tables+
    for df in [households,people,h_marg,p_marg]:
        for geog_cat in geog_id.keys():
            df[geog_cat] = geog_id[geog_cat]
    
    #synthesized households index is hh_id, assign it to bg_hh_id, also rename people hh_id to avoid confusion
    households['bg_hh_id']=households.index.values
    people.rename(columns={"hh_id":"bg_hh_id"}, inplace=True)

    key = BlockGroupID(
        geog_id['state'], geog_id['county'], geog_id['tract'],
        geog_id['block group'])
    fit_quality = FitQuality(people_chisq, people_p)

    return (households, people, [key,fit_quality], h_marg, p_marg)

def synthesize_one_wrap(args):
   return synthesize_one(*args)

def synthesize_all(recipe, num_geogs=None, indexes=None,
                   marginal_zero_sub=.01, jd_zero_sub=.001):
    """
    Returns
    -------
    households, people : pandas.DataFrame
    fit_quality : dict of FitQuality
        Keys are geographic IDs, values are namedtuples with attributes
        ``.household_chisq``, ``household_p``, ``people_chisq``,
        and ``people_p``.

    """
    print "Synthesizing at geog level: '{}' (number of geographies is {})".\
        format(recipe.get_geography_name(), recipe.get_num_geographies())
    
    proc = psutil.Process(os.getpid())
    print 'memory 1', proc.memory_info().rss
    if indexes is None:
        indexes = recipe.get_available_geography_ids()
    indexes=list(indexes)
    
    cnt = 0
    fit_quality = {}
    hh_index_start = 0

    #print 'pool exist', pool
    #multiprocess synthesis
    time0=time.time()
    #pool = mp.Pool(processes=6,maxtasksperchild=50)
    pool = mp.Pool(processes=8)
    #results = [pool.apply_async(synthesize_one, args=(recipe,geog_id)) for geog_id in indexes]
    results = pool.map(synthesize_one_wrap, [(recipe,geog_id) for geog_id in indexes])
    pool.close()
    pool.join()
    print "multiprocessing time in seconds: ",time.time()-time0
    proc = psutil.Process(os.getpid())
    print 'memory multiprocessing end', proc.memory_info().rss   
    #results=[res.get() for res in results]
    results=zip(*results)

    
    #process results
    all_households = pd.concat(results[0], ignore_index=True)
    all_households['hh_id']=all_households.index.values
    
    all_persons = pd.concat(results[1], ignore_index=True)
    joinid=list(geog_id.keys())+['bg_hh_id']
    all_persons = pd.merge(all_persons ,all_households[joinid+['hh_id']], on = joinid,how='left')
    
    fit_quality= dict((item[0],item[1]) for item in results[2])

    df_h_marg=pd.concat(results[3], ignore_index=True)
    df_p_marg=pd.concat(results[4], ignore_index=True)
    print "result produce time in seconds: ",time.time()-time0
    proc = psutil.Process(os.getpid())
    print 'memory result end', proc.memory_info().rss
    del results, pool
    mp.active_children()  # discard dead process objs
    gc.collect()  
    
    return (all_households, all_persons, fit_quality, df_h_marg,df_p_marg )
