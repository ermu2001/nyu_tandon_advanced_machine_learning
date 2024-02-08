
import itertools
import os
import os.path as osp
from datasets import (
    load_eye,
)
import tqdm
import numpy as np

import pickle as pkl

def normalize(data):
    data = (data - data.mean()) / data.std()
    return data



# cluster_method2func={
#     '': 
# }
def cluster(features, centers):
    # k means clustering
    # featrues: shaped (num_samples, feature_dim)
    # centers: shaped (num_centers, feature_dim)
    feature_dim = features.shape[-1]
    f = features[:, None, ...]
    c = centers[None, ...]
    distance = np.sum((f-c) ** 2, axis=-1) # shaped (num_samples, num_centers): each sample to each centers' corresponding l2 distance
    assignment = np.argmin(distance, axis=-1) # shaped (num_samples, ): each samples' assignment
    
    # update centers:
    new_centers = np.zeros_like(centers)
    for i in range(centers.shape[0]):
        assigned_features = features[assignment == i]
        if assigned_features.shape[0] == 0:
            new_center = np.random.randn(feature_dim)
        else:
            new_center = np.mean(assigned_features, axis=0)
        new_centers[i, ...] = new_center
    return new_centers, assignment

def compute_cost(features, centers, assignment):
    mean_costs, median_costs, center_costs = [], [], []
    for i in range(centers.shape[0]):
        assigned_features = features[assignment == i]
        if assigned_features.shape[0] == 0:
            continue
        center = centers[i][None, ...]
        mean_cost = np.sum(np.square(assigned_features-center), axis=-1).sum()
        median_cost = np.sqrt(np.sum(np.square(assigned_features-center), axis=-1)).sum()
        center_cost = np.sqrt(np.sum(np.square(assigned_features-center), axis=-1)).max()
        mean_costs.append(mean_cost)
        median_costs.append(median_cost)
        center_costs.append(center_cost)
    
    mean_cost = np.sum(mean_costs)
    median_cost = np.sum(median_costs)
    center_costs = np.max(center_costs)
    return mean_cost, median_cost, center_cost
    
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_eye()['train']
    features, classes = data[:, :-1].astype(np.float32), data[..., -1].astype(np.int8)
    num_samples, num_features = features.shape
    features = normalize(features) # normalize except for the classes 

    # old_centers = np.random.randn(args.num_classes, num_features)
    old_centers = features[[0, 1, -1, -2]]
    kmean_costs, kmedian_costs, kcenter_costs = [], [], []
    tbar = tqdm.trange(args.num_iters)
    for i in tbar:
        centers, assignment = cluster(features, old_centers)
        kmean_cost, kmedian_cost, kcenter_cost = compute_cost(features, old_centers, assignment)
        kmean_costs.append(kmean_cost)
        kmedian_costs.append(kmedian_cost)
        kcenter_costs.append(kcenter_cost)
        tbar.set_description_str(
            f'kmean_cost:{kmean_cost}; '
            f'kmedian_cost:{kmedian_cost}; '
            f'kcenter_cost:{kcenter_cost}; '
        )

        if np.all(np.isclose(old_centers, centers)):
            print('nearly no update in centers, will break')
            break
        else:
            old_centers = centers

    losses = {
        'kmean_costs': kmean_costs,
        'kmedian_costs': kmedian_costs,
        'kcenter_costs': kcenter_costs,
    }
    with open(osp.join(args.output_dir, 'losses.pkl'), 'wb') as f:
        pkl.dump(losses, f)



if __name__ == '__main__':
    from config import eye_config
    print(eye_config)
    main(eye_config)

