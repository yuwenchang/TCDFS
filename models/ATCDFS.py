"""Time:2025/7/31
Author:Yu wenchang
YU Wenchang 2025
Transitive-closure fuzzy rough sets and feature selection.
Apply closure per attribute, then divide by its maximum to obtain distances in [0,1]."""
import numpy as np
from util import *
import datetime
from conf import parse_args, dataconf
import logging
import math
from datasets import data_minmax_stand
import pandas as pd
import heapq
def get_single_distances(args, attributions, target_atrribute_list):
    """Compute per-attribute distances from an attributes-by-samples matrix.

    target_atrribute_list contains one decision label per sample and checks
    sample alignment. Categorical distances are 0/1; numeric distances are
    absolute differences. Return shape: (n_attributes, n_samples, n_samples).
    ATCDFS additionally closes and normalizes each numeric distance matrix."""
    attribute_num,sample_num = attributions.shape
    assert len(target_atrribute_list) == len(attributions[0]), "raw data sample num != target sample num"
    SADM = np.zeros((attribute_num, sample_num, sample_num), dtype=np.float64)# m n*n single attribute distance matrix

    if "label_attributes_index" in args.dataconf[args.dataname]:
        label_attributes_index = args.dataconf[args.dataname]["label_attributes_index"]
        args.logger.debug(label_attributes_index)
    for aindex, ai in enumerate(attributions):
        if aindex in label_attributes_index:
            # Different categorical values have distance 1.
            diff = (ai[:, np.newaxis] != ai).astype(np.float64)
            SADM[aindex] = diff
        else:
            # Numeric distance is the absolute difference.
            diff = np.abs(ai[:, np.newaxis] - ai)
            attri_tcd_dist = get_transitive_closure_distance(args,diff)
            if np.max(attri_tcd_dist)==0: #same value
                SADM[aindex] = attri_tcd_dist
            else:
                SADM[aindex] = attri_tcd_dist/np.max(attri_tcd_dist)

    return  SADM

def get_multi_attribute_distance(args, cur_distance_matrix,
                                 single_attribute_distance_matrix,
                                 add_attribute_index):
    """Add one attribute to the current Euclidean distance matrix.

    cur_distance_matrix has shape (n, n); single_attribute_distance_matrix
    has shape (m, n, n). add_attribute_index is an index in [0, m).
    Return the updated (n, n) pairwise distance matrix."""
    multi_attribute_distance_matrix = np.sqrt(np.square(np.array(cur_distance_matrix)) + np.square(np.array(
        single_attribute_distance_matrix[add_attribute_index])))

    return multi_attribute_distance_matrix

def get_transitive_closure_distance(args,dist_matrix):
    """Algorithm 1: return the full min-max transitive closure."""
    current = np.copy(dist_matrix)
    n = len(current)
    # For n >= 2, bit_length equals floor(log2(n - 1)) + 1.
    for _ in range(max(n - 1, 0).bit_length()):
        squared = np.empty_like(current)
        for x in range(n):
            # Synchronous composition: minimize max(d[x,p], d[p,y]) over p.
            squared[x] = np.maximum(current[x, :, None], current).min(axis=0)
        if np.array_equal(squared, current):
            break
        current = squared
    return current


def get_neighbors_by_distance(args, distance_matrix):
    """Return neighbor-index lists for an (n_samples, n_samples) distance matrix.

    args.model selects a delta or k neighborhood. args.tcdfs_delta is the
    radius; args.tcdfs_k is the rank cutoff (including self and boundary ties)."""
    n_samples = distance_matrix.shape[0]
    neighbors = []

    if args.model == "atcdfs_delta":
        # Use samples within tcdfs_delta as neighbors.
        for i in range(n_samples):
            # Get distances from sample i.
            distances = distance_matrix[i]
            # Find indices with distance below tcdfs_delta.
            neighbor_indices = np.where(distances < args.tcdfs_delta)[0]
            # Exclude the sample itself.
            neighbor_indices = neighbor_indices[neighbor_indices != i].tolist()
            neighbors.append(neighbor_indices)

    elif args.model == "atcdfs_k":
        # Include neighbors through rank k, including self and boundary ties.
        for i in range(n_samples):
            dist = distance_matrix[i]
            order = np.argsort(dist)  #
            d_cut = dist[order[args.tcdfs_k - 1]]  #
            last = np.searchsorted(dist[order], d_cut + 1e-12, side='right')
            neighbor_indices = order[:last].tolist()
            neighbors.append(neighbor_indices)

    else:
        raise ValueError("Invalid model type. Supported models are 'tcdfs_delta' and 'tcdfs_k'.")

    return neighbors


def get_lower_approximation(args,neighbor_list,Di,Vi):
    """Compute the lower approximation of decision class Di.

    Vi contains candidates outside the positive region (Di minus Pi).
    Return the updated positive region and remaining candidates."""
    new_postive_set = set()
    old_postive_set = set(Di)-set(Vi)
    for sample_index in Vi:
        if set(neighbor_list[sample_index]).issubset(set(Di)):
            new_postive_set.add(sample_index)
    new_V = set(Vi) -  new_postive_set
    return new_postive_set|old_postive_set,new_V

def get_dependency(args,neighbor_list,D_partitions_list,V_list):
    """Compute dependency from neighborhoods and decision partitions.

    D_partitions_list contains the decision classes; V_list contains their
    candidates outside the positive region. Return dependency and updated V_list."""
    sample_num = len(neighbor_list)
    postive_sample_num = 0
    new_V_list = []
    for index,Di in enumerate(D_partitions_list):
        postive_sample,new_Vi = get_lower_approximation(args,neighbor_list,Di,V_list[index])
        postive_sample_num += len(postive_sample)
        new_V_list.append(new_Vi)
    dependency = 1.0*postive_sample_num/sample_num
    return dependency,new_V_list
def get_target_partition_list(args,target_label_list):
    label_dict = {}
    for idx, label in enumerate(target_label_list):
        if label in label_dict:
            label_dict[label].append(idx)
        else:
            label_dict[label] = [idx]
    return list(label_dict.values())
def feature_selelction_base_on_atcdfs(args, data, target_label_list):
    starttime = datetime.datetime.now()
    sample_num = len(target_label_list)
    stand_data = np.array(data_minmax_stand(args,pd.DataFrame(np.array(data).tolist())))
    single_attribute_distance_matrix = get_single_distances(args, stand_data, target_label_list)
    target_partition_list = get_target_partition_list(args,target_label_list)
    V_list = target_partition_list.copy()
    red = []
    red_distance = [[0 for i in range(sample_num)] for i in range(sample_num)]  # Initialize all pairwise distances to zero.
    candi_attribute_index_list = [i for i in range(len(data))]
    dependency = -1
    args.logger.info("reduction start")
    while len(candi_attribute_index_list)>0:
        candi_best_attribute_index = -1
        candi_best_dependency = dependency
        candi_best_red_distance = red_distance
        candi_best_V_list = []
        for index,c in enumerate(candi_attribute_index_list):
            transitive_candi_distance = get_multi_attribute_distance(args,red_distance,single_attribute_distance_matrix,c)
            dist_neighbor_list = get_neighbors_by_distance(args,transitive_candi_distance)
            dependency,new_V_list = get_dependency(args,dist_neighbor_list,target_partition_list,V_list)
            args.logger.debug("{}U{} dependecy={}".format(red,[c],dependency))
            if(candi_best_dependency < dependency):
                candi_best_dependency = dependency
                candi_best_attribute_index = c
                candi_best_V_list = new_V_list
                candi_best_red_distance = transitive_candi_distance
        if(candi_best_attribute_index==-1): #No candidate increases dependency.
            args.logger.info("reduction end no attribute make dependecy larger")
            break
        else:          #Add the candidate with the highest dependency to red.
            args.logger.info("{}U{}-->{} dependency={}".format(red,[candi_best_attribute_index],red+[candi_best_attribute_index],candi_best_dependency))
            red = red + [candi_best_attribute_index]
            red_distance = candi_best_red_distance
            candi_attribute_index_list.remove(candi_best_attribute_index)
            dependency = candi_best_dependency
            V_list = candi_best_V_list
            if dependency == 1:
                break
    endtime = datetime.datetime.now()
    args.logger.info("reduction end,cost time = {}".format((endtime-starttime).total_seconds()))
    return red

def test_feature_selelction_base_on_atcdfs(args):
    np.set_printoptions(precision=2)
    raw_data = np.array([[0.0, 0.17, 0.56, 0.63, 1.0],
           [0.1, 0.14, 0.94, 0.98, 0.25],
           [0.21, 0.0, 0.56, 0.33, 0.88],
           [0.15, 1.0, 0.83, 0.75, 0.25]])
    D = [1, 1, 2, 2, 2]
    sadm = get_single_distances(args, raw_data,D)
    td = get_target_partition_list(args,D)
    print(sadm)
    print(td)

    attribute_num,sample_num = raw_data.shape
    red_distance = np.full((sample_num,sample_num),0).astype(np.float64)
    # Test multi-attribute distance computation.
    cur_distance = get_multi_attribute_distance(args,red_distance,sadm,0)
    cur_distance = get_multi_attribute_distance(args,cur_distance, sadm, 1)
    print(cur_distance)
    # Test neighborhood construction from distances.
    D_neighbors = get_neighbors_by_distance(args, cur_distance)
    print(D_neighbors)

    # Test feature selection using the neighborhood relation.
    feature_selelction_base_on_atcdfs(args, np.array(raw_data), D)



if __name__=="__main__":

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    args = parse_args()
    args.logger = logger
    args.dataconf = dataconf()
    print(args.logger.getEffectiveLevel())
    args.logger.setLevel(logging.DEBUG)
    print(args.logger.getEffectiveLevel())
    test_feature_selelction_base_on_atcdfs(args)