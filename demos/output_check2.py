
# coding: utf-8
#output check
#multiprocessing version

import pandas as pd
import os
import sys
from scipy.stats import chisquare

def compare_to_margin(synout,margin):

    # remove any items that are zero in the constraints   
    synout, margin = synout.align(margin)
    w = margin >= 1
    synout, margin = synout[w], margin[w]
    return chisquare(synout.values, margin.values)

def get_chisq_table(df_summary, gplevel, col1, col2):
    #get chi square by col1 and col2 for specific geographic units(gplevel)
    lst_com=[]
    for name, dfgroup in df_summary.groupby(gplevel):
        lst_com.append(name+compare_to_margin(dfgroup[col2],dfgroup[col1]))
    df_compare= pd.DataFrame(lst_com)
    df_compare.columns=[gplevel+['stats','p_value']]
   
    return df_compare

def get_sum_table_by_grp(df,grp,colname):
    #df, dataframe; grp, groups
    #df.set_index(grp, inplace=True)
    dfsum=df.groupby(grp).sum()
    print dfsum
    dfsum.columns=[colname]
    return dfsum.reset_index(inplace=True) 
    
def output_check(dict_in):
    # dict_in {'households':[hh_marginal,households], 'persons':[pp_marginal,persons]}
    dict_out={}
    geos=['state','county','tract','block group']
    geos_cat=geos+['cat_name']
    df_marginals={}

    for key in  dict_in.keys():
        df_marg_bgcv=dict_in[key][0].reset_index()
        df_output=dict_in[key][1]

        cats=df_marg_bgcv['cat_name'].unique()
        #sum output by individual marginal category and combine the results
        dfgroup_lst=[]
        for cat in cats:
            dfgroup=df_output.groupby(geos+[cat]).size()
            dfgroup=pd.DataFrame(dfgroup,columns=[key]).reset_index()
            dfgroup['cat_name']=cat
            dfgroup.rename(columns={cat: 'cat_value'}, inplace=True)
            dfgroup_lst.append(dfgroup)
        dfgall=pd.concat(dfgroup_lst)
        dfgall.reset_index(inplace=True)

        #join outout summary table to marginal table for comparison
        df_bgcv=pd.merge(df_marg_bgcv, dfgall, on=geos_cat+['cat_value'], how='left')
        df_chi_bgc=get_chisq_table(df_bgcv, geos_cat,'marginal', key)
        
        df_bgc=df_bgcv.groupby(geos_cat).agg({'marginal': 'sum', key:'sum'}) #return dataframe
        df_bgc.rename({'marginal':'marginal_total', key: key+'_total'})

        #df_bgcv.set_index(geos_cat,inplace=True)
        for df in [df_bgcv, df_bgc]:
            df['dif']=df[key]-df['marginal']
            df['dif%']=df['dif']/df['marginal']*100.0

        df_bgc=pd.merge(df_bgc.reset_index(),df_chi_bgc, on=geos_cat, how='left')
        dict_out[key]=[df_bgc, df_bgcv]

    #hh_size analysis
    df_marg, households,persons = \
        dict_out['households'][1].reset_index(), \
        dict_in['households'][1].reset_index(), \
        dict_in['persons'][1].reset_index()
    df_marg_size=df_marg[df_marg['cat_name']=='hh_size']
    df_marg_size.rename(columns={'cat_value':'hh_size'},inplace=True)
    
    df_output=pd.merge(persons,households[['hh_id','hh_size']],on='hh_id', how='left')
    sum_size=df_output.groupby(geos+['hh_size']).size()
    df_marg_size.set_index(geos+['hh_size'],inplace=True)
    df_marg_size['persons']=sum_size
    df_marg_size.reset_index(inplace=True)
    df_marg_size.fillna(value=0, inplace=True)
    dict_out['hhsize']=df_marg_size


    dfhh7size=df_marg_size[df_marg_size['hh_size']=='seven or more']
    dfhh7size['hh_size_weight_7']=dfhh7size['persons']/dfhh7size['households']
    dfhh7size.fillna(value=0, inplace=True)
    dict_out['hhsize7plus']=dfhh7size
    
    totalpersons=dict_out['households'][0].query('cat_name=="' + cats[0] + '"')['marginal'].sum()/persons.shape[0]
    df_ttlpersons=pd.DataFrame(data=[totalpersons],columns=['hh_size_person_factor'])
    dict_out['person_factor']=df_ttlpersons

    return dict_out
   
