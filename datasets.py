import pandas
import pandas as pd
import numpy as np
from conf import *
import logging
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler




def data_stand1(data,axis=1):
    """Min-max scale features; axis=1 means features in rows, axis=0 in columns."""
    scaler = MinMaxScaler()
    if axis == 1:
        df_normalized = pd.DataFrame(scaler.fit_transform(data.T).T, columns=data.columns)
    else:
        df_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return df_normalized
def data_minmax_stand(args,data,axis=1):
    categorical = args.dataconf[args.dataname]["label_attributes_index"] or []
    frame = data.T.copy() if axis == 1 else data.copy()
    numeric = [i for i in range(frame.shape[1]) if i not in categorical]
    if numeric:
        frame.iloc[:, numeric] = MinMaxScaler().fit_transform(frame.iloc[:, numeric].astype(float))
    return frame.T if axis == 1 else frame

def data_stand2(data,axis=1):
    mean = data.mean(axis=axis)
    std = np.sqrt(data.var(axis=axis))
    return data.sub(mean,axis=1-axis).div(std,axis=1-axis) #Standardize values to zero mean and unit standard deviation.

#(I, O)-Fuzzy Rough Sets Based on Overlap Functions With Their Applications to Feature Selection and Image Edge Extraction
#Normalization method described in the paper above.
def data_stand3(args,matrix):
    label_attributes_index = args.dataconf[args.dataname]["label_attributes_index"]
    processed_matrix = []
    for index,row in enumerate(np.array(matrix)):
        if index in label_attributes_index:
            processed_matrix.append(row) #Categorical attributes do not require normalization.
            continue
        # Compute the sum of squared values in each row.
        sum_of_squares = np.sum(np.square(row))
        # Compute the Euclidean norm.
        root_sum_of_squares = np.sqrt(sum_of_squares)
        # Divide each value by the row norm.
        processed_row = row / root_sum_of_squares
        processed_matrix.append(processed_row)
    return processed_matrix
def data_stand4(matrix):
    processed_matrix = []
    for row in np.array(matrix):
        # Compute the maximum of each row.
        max = np.max(row)
        # Scale by the row maximum.
        processed_row = row / max
        processed_matrix.append(processed_row)
    return processed_matrix

def get_decision(labels):
    d = {}
    for index,label in enumerate(labels):
        if label in d:
            d[label].append(index)
        else:
            d[label] = [index]
    return d
def load_data(args):
    args.logger.info("load data start, datapath {}".format(args.dataconf[args.dataname]["filepath"]))
    datalist = []
    colindex = None
    na_attribute_index_list = []
    labels = []
    nlabel_cond_attri_idx_list = []
    path = Path(args.dataconf[args.dataname]["filepath"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    for lindex,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines()):
        if not line.strip():
            continue
        toks = [key.strip() for key in line.split(args.dataconf[args.dataname]["seperator"])]

        if args.dataconf[args.dataname]["header"] == lindex:
            continue
        if args.dataname=="promoters":
            toks = [toks[0].strip()]+list(toks[-1].strip())
        format_toks = []

        for index,item in enumerate(toks):
            if index in args.dataconf[args.dataname]["remove_attribute_index"]: #Exclude identifiers and row numbers from the features.
                continue
            if index == args.dataconf[args.dataname]["label_index"]:
                labels.append(str(item))
                continue
            if index in args.dataconf[args.dataname]["ori_lab_condattri_idx"]:
                format_toks.append(str(item))
                nlabel_cond_attri_idx = len(format_toks)-1
                if nlabel_cond_attri_idx not in nlabel_cond_attri_idx_list:
                    nlabel_cond_attri_idx_list.append(nlabel_cond_attri_idx)
            else:
                if item.lower() in {"?", "na", "nan", ""}:
                    format_toks.append(np.nan)
                else:
                    format_toks.append(float(item))
        datalist.append(format_toks)
    args.dataconf[args.dataname]["label_attributes_index"] = nlabel_cond_attri_idx_list
    args.logger.info("new label conditional attribute idx: {}".format(nlabel_cond_attri_idx_list))
    # Leave missing numeric values for training-fold imputation.
    df = pd.DataFrame(datalist,columns=colindex)
    dec = pandas.Series(labels)
    df = df.T

    args.logger.info("df shape:"+str(df.shape)+" label num:"+str(dec.shape))
    args.logger.info("classes num: "+str(len(set(dec)))+", instance num of each label: "+str({k:dec.tolist().count(k) for k in set(dec)}))
    args.logger.info("instance num: "+str(len(df.columns))+", attribute num: "+str(len(df.index)))
    args.logger.info("load data end ")
    return df,dec

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(message)s',
                        level=logging.INFO  )
    logger = logging.getLogger(__name__)
    args = parse_args()
    args.dataconf = dataconf()
    args.logger = logger

    datalist = ["iris", "liverdisorder", "glass", "forestfire", "vehicle", "german", "parkinsons",
                "wdbc", "promoters","sonar", "movement", "muskv1", "colonTumor", "DLBCLStanford", "AMLALL","DLBCLNIH"]
    for dn in datalist:
        args.dataconf = dataconf()
        args.dataname = dn
        X, y = load_data(args)

