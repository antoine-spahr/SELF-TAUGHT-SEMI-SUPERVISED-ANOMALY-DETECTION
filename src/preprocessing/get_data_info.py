#import numpy as np
import pandas as pd
#import os
#import sys
#import matplotlib.pyplot as plt

#DATA_PATH = r'../../../data/'

def generate_data_info(csv_fn):
    """
    Extract the information by patient and body part with the filename to any
    image of the MURA-v1.1 dataset to facilitate the access and sorting.
    ----------
    INPUT
        |---- csv_fn (str) the filename of the csv of filename
    OUTPUT
        |---- df (pandas.dataframe) of image filename with additional information
        |         to facilitate access of data : if a patient has any abnormal XR
        |         or if a patient has any abnormal XR for a given body part.
    """
    df_fn = pd.read_csv(csv_fn, header=None, names=['filename'])
    df_info = pd.read_csv(csv_fn, \
                          header=None, names=['body_part','patientID','study'], \
                          sep='/', usecols=[2,3,4])
    df = pd.concat([df_info, df_fn], axis=1)
    # remove MURA-v1.1/train or MURA-v1.1/valid
    df.filename = df.filename.apply(lambda s: s[16:])
    # remove XR_
    df.body_part = df.body_part.apply(lambda s: s[3:])
    # create the label : 1 if positive, else 0
    df['label'] = df.study.apply(lambda s: 0 if s[7:] == 'negative' else 1)
    df.drop(columns={'study'}, inplace=True)
    # get if a patient has only/no/some abnormal xray for a given body part
    df_abnormal_bp = df[['patientID','body_part','label']].groupby(['patientID','body_part']).mean() \
                                                          .reset_index()
    df_abnormal_bp.label = df_abnormal_bp.label.apply(lambda x: 0.5 if ((x>0) and (x<1)) else x)
    # get if a patient has an abnormal xray on any body part
    df_abnormal_all = df[['patientID','label']].groupby('patientID').mean()
    df_abnormal_all.label = df_abnormal_all.label.apply(lambda x: 0.5 if ((x>0) and (x<1)) else x)
    # merge with the whole dataframe
    df = pd.merge(df, df_abnormal_all, how='outer', left_on='patientID', right_index=True) \
           .rename(columns={'label_x':'abnormal_XR', 'label_y':'patient_any_abnormal'})
    df = pd.merge(df, df_abnormal_bp, how='outer', left_on=['patientID','body_part'], right_on=['patientID','body_part']) \
           .rename(columns={'label':'body_part_abnormal'})

    return df

# %%
# df1 = generate_data_info(DATA_PATH+'RAW/train_image_paths.csv')
# df2 = generate_data_info(DATA_PATH+'RAW/valid_image_paths.csv')
# df = pd.concat([df1,df2], axis=0).reset_index()
# df.to_csv(DATA_PATH+'data_info.csv')
