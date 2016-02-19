
# coding: utf-8

from synthpop.recipes.starter import Starter
from synthpop.synthesizer import synthesize_all
import pandas as pd
import os
import sys
def output_verify(households,persons,marginals):
    #marginals are list of marginal df by BG, odd number dfs are HH marg; even number dfs are person marg
    mlst={'households':households,'persons':persons}
    geos=['state','county','tract','block group']
    cat_grp=geos+['cat_name']
    df_marginals={}
    for key in mlst.keys():
        dfm=pd.concat(marginals[mlst.keys().index(key)::2])
        results=dfm.groupby(cat_grp).sum()
        dfm.set_index(cat_grp, inplace=True)
        dfm['marg_total']=results
        df_marginals[key]=dfm.reset_index()

    verilist=[]
    for key in mlst.keys():
        df_marg=df_marginals[key]
        cats=df_marg['cat_name'].unique()
        if key=='persons':
            df_output=pd.merge(persons,households,left_on='hh_id',right_index=True, how='left') 
        else:
            df_output=households 
        
        #sum output by individual marginal category and combine the results
        dfgroup_lst=[]
        for cat in cats:
            dfgroup=df_output.groupby(geos+[cat]).size()
            dfgroup=pd.DataFrame(dfgroup,columns=['output_sub']).reset_index()
            dfgroup['cat_name']=cat
            dfgroup.rename(columns={cat: 'cat_value'}, inplace=True)
            dfgroup_lst.append(dfgroup)
        dfgall=pd.concat(dfgroup_lst)

        #join outout summary table to marginal table for comparison
        dfmerge=pd.merge(df_marg,dfgall,on=cat_grp+['cat_value'], how='left')
        dfmerge.set_index(cat_grp,inplace=True)
        dfgp2=dfmerge.groupby(level=cat_grp)['output_sub'].sum()
        dfmerge['output_total']=dfgp2
        dfmerge['dif']=dfmerge['output_total']-dfmerge['marg_total']
        dfmerge['dif%']=dfmerge['dif']/dfmerge['marg_total']*100.0
        verilist.append(dfmerge)

        #for persons output, join distribution by hh_size for further analysis
        if key=='persons':
            df_marg=df_marginals['households']
            mg_size=df_marg[df_marg['cat_name']=='hh_size']
            mg_size.rename(columns={'cat_value':'hh_size'},inplace=True)
            syn_size=df_output.groupby(geos+['hh_size']).size()
            df_syn_size=pd.DataFrame(syn_size,columns=['output_persons']).reset_index()
            merge_size=pd.merge(mg_size,df_syn_size,on=geos+['hh_size'],how='left')
            verilist.append(merge_size)

    return verilist
   
