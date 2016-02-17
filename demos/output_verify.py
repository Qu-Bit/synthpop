
# coding: utf-8

from synthpop.recipes.starter import Starter
from synthpop.synthesizer import synthesize_all
import pandas as pd
import os
import sys
def output_verify(households,persons,marginals):
    mlst=['households','persons']
    geos=['state','county','tract','block group']
    cat_grp=geos+['cat_name']
    df_marginals={}
    for i in [0,1]:
        dfm=pd.concat(marginals[i::2])
        results=dfm.groupby(cat_grp).sum()
        dfm.set_index(cat_grp, inplace=True)
        dfm['marg_total']=results
        df_marginals[mlst[i]]=dfm.reset_index()

    verilist=[]    
    for key in mlst:

        if key=='households':
            df_marg=df_marginals[key]
            cats=df_marg['cat_name'].unique()
            df_output=households #read synthesized HHs  
        else:
            df_marg=pd.concat(df_marginals.values())
            cats=df_marg['cat_name'].unique()
            df_output=pd.merge(persons,households,left_on='hh_id',right_index=True, how='left') 
        
        dfgroup_lst=[]
        for cat in cats:
            dfgroup=df_output.groupby(geos+[cat]).size()
            dfgroup=pd.DataFrame(dfgroup,columns=['output_sub']).reset_index()
            dfgroup['cat_name']=cat
            dfgroup.rename(columns={cat: 'cat_value'}, inplace=True)
            dfgroup_lst.append(dfgroup)
        dfgall=pd.concat(dfgroup_lst)
        dfmerge=pd.merge(df_marg,dfgall,on=cat_grp+['cat_value'], how='left')

        dfmerge.set_index(cat_grp,inplace=True)
        dfgp2=dfmerge.groupby(level=cat_grp)['output_sub'].sum()
        dfmerge['output_total']=dfgp2
        verilist.append(dfmerge)

    return verilist
   
