import os
import pandas as pd
import numpy
import synthesizer_algorithm.adjusting_sample_joint_distribution
import synthesizer_algorithm.drawing_households
import synthesizer_algorithm.pseudo_sparse_matrix
import time
from scipy import sparse

# this function adds the unique_ids that were previously added by adjusting_sample_joint_distribution.create_update_string and add_unique_id
def prepare_data(data_dir, hh_sample_file, per_sample_file, hh_marginals_file, per_marginals_file):
    os.chdir(data_dir)
    hh_sample = pd.read_csv(hh_sample_file, header = 0)
    hh_sample = hh_sample.astype('int')
    hh_vars = np.array((hh_sample.columns)[4:]) # identifies the household control variables
    hh_var_list = list(hh_sample.columns[4:])
    hh_dims = np.array((hh_sample.max())[4:]) # identifies number of categories per household control variable
    hhld_units = len(hh_sample.index)  # identifies the number of housing units to build the Master Matrix
    hh_sample['group_id'] = ''
    for var in hh_var_list:
        hh_sample[var + '_str'] = hh_sample[var].astype('str')
        hh_sample.group_id = hh_sample.group_id + hh_sample[var + '_str']
        hh_sample = hh_sample.drop([var + '_str'], axis=1)
    hh_marginals = pd.read_csv(hh_marginals_file, header = 0)
    hhid = hh_sample.groupby(['group_id'], as_index=False)['state'].min()
    hhid['hhld_uniqueid'] = hhid.index + 1
    hhid = hhid[['group_id', 'hhld_uniqueid']]
    hh_sample = pd.merge(hh_sample, hhid, how='left', left_on='group_id', right_on='group_id')
    hh_sample = hh_sample.drop('group_id', axis=1)
    per_sample = pd.read_csv(per_sample_file, header = 0)
    per_vars = list(per_sample.columns)[5:] # identifies the person control variables
    per_dims = np.array(per_sample.astype('int').max())[5:] # identifies number of categories per household control variable
    per_vars_dims = dict(zip(per_vars, per_dims))
    per_marginals = pd.read_csv(per_marginals_file, header = 0)
    
    matrix = populate_master_matrix(hh_dims, per_dims, hhld_units, hh_sample)
    sparse_matrix = pseudo_sparse_matrix(data_dir, hh_sample)
    index_matrix = generate_index_matrix(sparse_matrix)
    
    housing_synthetic_data = pd.DataFrame(columns=['state','county','tract','bg','hhid','serialno','frequency','hhuniqueid'])
    person_synthetic_data = pd.DataFrame(columns=['state','county','tract','bg','hhid','serialno','pnum','frequency','personuniqueid'])
    
    performance_statistics = pd.DataFrame(columns=['state','county','tract','bg','chivalue','pvalue','synpopiter','heuriter','aardvalue'])

    return {'matrix':matrix, 'sparse_matrix':sparse_matrix, 'index_matrix':index_matrix, 'housing_synthetic_data':housing_synthetic_data, 'person_synthetic_data':person_synthetic_data}

