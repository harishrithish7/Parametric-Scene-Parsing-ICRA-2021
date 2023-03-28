import numpy as np
import os
from tqdm import tqdm
import math
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from model_combined_branch import CombinedNet
from model_ground_branch import GroundNet
from model_aerial_branch import AerialNet

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

torch.set_printoptions(precision=7)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default=None, help="path to saved checkpoint file", required=True)
parser.add_argument("--lookahead", type=int, default=None, help="lookahead distance", required=True)
parser.add_argument("--inputs", type=str, default=None, help="aerial, combined or sv", required=True)
parser.add_argument("--batch_size", type=int, default=12, help="Batch size", required=False)
parser.add_argument("--datadir", type=str, default=None, help="Path to the dataset directory", required=True)
parser.add_argument("--dataset", type=str, default=None, help="kitti or argo dataset", required=True)
opt = parser.parse_args()
print(opt, flush=True)

cuda = torch.cuda.is_available()
print ("Using cuda:", cuda, flush=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###########################################################################
#                       Options & Hyperparameters
###########################################################################
lookahead_distance = opt.lookahead
aerial_range = lookahead_distance + 10
rotate_aerial = True
num_workers = 8
batch_size = opt.batch_size

num_classes = {}
if opt.dataset == "kitti":
    from datasets_kitti import CombinedDataset

    num_clusters = {
        "discrete": [7,3]
    }
    num_classes['binary'] = 14
    num_classes['continuous'] = 9
    num_classes['discrete'] = 2

elif opt.dataset == "argo":
    from datasets_argo import CombinedDataset

    num_clusters = {
        "discrete": [6,3]
    }
    num_classes['binary'] = 8
    num_classes['continuous'] = 2
    num_classes['discrete'] = 2

num_classes['total'] = num_classes['binary'] + num_classes['continuous'] + num_classes['discrete']

hparams = {
    "inputs": opt.inputs,
    "lookahead": lookahead_distance,
    "aerial_range": aerial_range,
    "rotate_aerial": rotate_aerial,
}
print (hparams, flush = True)

# mean and standard deviation of ground imagery (train set)
cf_mean = torch.Tensor([0.47226224, 0.47520779, 0.51034265])
cf_std = torch.Tensor([0.20337067, 0.21041222, 0.22835514])

# mean and standard deviation of aerial imagery (train set)
aerial_mean = torch.Tensor([0.42290114, 0.41879349, 0.41141361])
aerial_std = torch.Tensor([0.22428898, 0.22103066, 0.21567185])

datadir =  opt.datadir

###########################################################################
#                          Model Initialization
###########################################################################
# Creating model instance
if opt.inputs == "combined":
    model = CombinedNet(num_classes, num_clusters, branch_layer="layer8", dataset=opt.dataset)
elif opt.inputs == "aerial":
    model = AerialNet(num_classes, num_clusters, branch_layer="layer8", dataset=opt.dataset)
elif opt.inputs == "sv":
    model = GroundNet(num_classes, num_clusters, branch_layer="layer8", dataset=opt.dataset)

# Data parallelization of model
print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
model = nn.DataParallel(model)

###########################################################################
#                      Load Model 
###########################################################################
checkpoint = torch.load(opt.checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

model.train()
if cuda:
    model.cuda()

###########################################################################
#                           Dataloader
###########################################################################
val_dataset = CombinedDataset(num_clusters, datadir, aerial_range, lookahead_distance, rotate_aerial, aerial_mean, aerial_std, cf_mean, cf_std, num_ground=1, mode="val")
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        pin_memory=True)    

attribute_names = val_dataset.attributes['names']
###########################################################################
#                           Metric Functions
###########################################################################
def compute_binary_results(metrics, mode, attribute_names):
    accuracies = []
    for i,var in enumerate(attribute_names["binary"]):
        tn, fp, fn, tp = metrics[mode]["binary"][var]["confusion_matrix"].ravel()
        # pos label as 0
        precision = tn / (tn + fn)
        recall = tn / (tn + fp)
        f1 = 2 * (precision * recall) / (precision + recall)
        metrics[mode]["binary"][var][0] = {"precision": precision, "recall": recall, "f1": f1}

        # pos label as 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        metrics[mode]["binary"][var][1] = {"precision": precision, "recall": recall, "f1": f1}      

        accuracy = (tp+tn) / (tp+tn+fp+fn)
        metrics[mode]["binary"][var]["accuracy"] = accuracy
        accuracies.append(accuracy)

    metrics[mode]["binary"]["accuracy"] = np.mean(accuracies)

def compute_continuous_results(metrics, mode, attribute_names, num_samples):
    metrics[mode]["continuous"]["mse"] = 0.
    metrics[mode]["continuous"]["mae"] = 0.
    for var in attribute_names["continuous"]:
        if num_samples[var]:
            metrics[mode]["continuous"]["mse"] += metrics[mode]["continuous"][var]["mse"].item()
            metrics[mode]["continuous"]["mae"] += metrics[mode]["continuous"][var]["mae"].item()

            metrics[mode]["continuous"][var]["mse"] = (metrics[mode]["continuous"][var]["mse"]/num_samples[var]).item()
            metrics[mode]["continuous"][var]["mae"] = (metrics[mode]["continuous"][var]["mae"]/num_samples[var]).item()

    if num_samples["total"]:
        metrics[mode]["continuous"]["mse"] = metrics[mode]["continuous"]["mse"]/num_samples["total"]
        metrics[mode]["continuous"]["mae"] = metrics[mode]["continuous"]["mae"]/num_samples["total"]
    
def compute_discrete_results(metrics, mode, attribute_names):
    discrete_accuracies = []
    for i,var in enumerate(attribute_names["discrete"]):
        var_accuracy = 0.
        for cls in range(metrics[mode]["discrete"][var]["ml_confusion_matrix"].shape[0]):
            tn, fp, fn, tp = metrics[mode]["discrete"][var]["ml_confusion_matrix"][cls].ravel()

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            accuracy = (tp+tn) / (tp+tn+fp+fn)
            metrics[mode]["discrete"][var][cls] = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}      

            var_accuracy += tp

        metrics[mode]["discrete"][var]["accuracy"] = var_accuracy / np.sum(metrics[mode]["discrete"][var]["ml_confusion_matrix"][0])
        discrete_accuracies.append(metrics[mode]["discrete"][var]["accuracy"])

    metrics[mode]["discrete"]["accuracy"] = np.mean(discrete_accuracies)

def compute_results(metrics, mode, attribute_names, num_samples):
    compute_binary_results(metrics, mode, attribute_names)
    compute_continuous_results(metrics, mode, attribute_names, num_samples["continuous"])
    compute_discrete_results(metrics, mode, attribute_names)
    
metrics = {
    "val": {
        "binary": {},
        "continuous": {},
        "discrete": {}
    }, 
}

##################################################
#               Initialize metrics
##################################################
for i, var in enumerate(attribute_names["binary"]):
    metrics["val"]["binary"][var] = {"confusion_matrix": np.zeros((2,2))}

for i, var in enumerate(attribute_names["continuous"]):
    metrics["val"]["continuous"][var] = {"mse": 0., "mae":0.}

for i, var in enumerate(attribute_names["discrete"]):
    num = num_clusters["discrete"][i]
    for mode in ["val"]:
        metrics[mode]["discrete"][var] = {"ml_confusion_matrix": np.zeros((num,2,2)), "confusion_matrix": np.zeros((num,num))}

#######################################################################################################
#                                           Validation
#######################################################################################################
num_samples_val = {
    "binary": 0.,
    "continuous": {},
    "discrete": 0.
}
for i, var in enumerate(attribute_names["continuous"]):
    num_samples_val["continuous"][var] = 0.
num_samples_val["continuous"]["total"] = 0.

val_batch = 0

pbar_val = tqdm(val_loader)
for aerial_img_loc, aerial_img, sv_locs, sv_img, binary_gt, binary_mask, continuous_gt, continuous_mask, discrete_gt, discrete_mask, meta in pbar_val:
    with torch.no_grad():
        # load data to GPU
        aerial_img_loc, aerial_img, sv_locs, sv_img, binary_gt, binary_mask, continuous_gt, continuous_mask, discrete_gt, discrete_mask = aerial_img_loc.to(device), aerial_img.to(device), sv_locs.to(device), sv_img.to(device), binary_gt.to(device), binary_mask.to(device), continuous_gt.to(device), continuous_mask.to(device), discrete_gt.to(device), discrete_mask.to(device)

        # Forward pass
        if opt.inputs == "combined":
            pred = model(aerial_img, sv_img)
        elif opt.inputs == "aerial":
            pred = model(aerial_img)
        elif opt.inputs == "sv":
            pred = model(sv_img)

        ##############################
        # binary
        ##############################
        for i,var in enumerate(attribute_names["binary"]):
            if torch.sum(binary_mask[:,i]) != 0:
                smax_binary = nn.functional.softmax(pred['binary'][i])
                pred_prob_binary, pred_class_binary = torch.max(smax_binary, 1)
                metrics["val"]["binary"][var]["confusion_matrix"] += confusion_matrix(binary_gt[:,i][binary_mask[:,i].bool()].cpu(), pred_class_binary[binary_mask[:,i].bool()].cpu(), labels=range(2))

        #############################
        # continuous
        #############################
        # metric
        for i, var in enumerate(attribute_names["continuous"]):
            if torch.sum(continuous_mask[:,i]) != 0:
                sq_err = torch.sum((pred['continuous'][i][continuous_mask[:,i].bool(),:]-continuous_gt[continuous_mask[:,i].bool(),i].view(-1,1)).pow(2))
                abs_err = torch.sum(torch.abs(pred['continuous'][i][continuous_mask[:,i].bool(),:]-continuous_gt[continuous_mask[:,i].bool(),i].view(-1,1)))
                metrics["val"]["continuous"][var]["mse"] += sq_err
                metrics["val"]["continuous"][var]["mae"] += abs_err
                
                num_samples_val["continuous"][var] += torch.sum(continuous_mask[:,i])
                    
        num_samples_val["continuous"]["total"] += torch.sum(continuous_mask).item()

        ##############################
        # discrete
        ##############################
        # metric
        for i,var in enumerate(attribute_names["discrete"]):
            if torch.sum(discrete_mask[:,i]) != 0:
                smax_discrete = nn.functional.softmax(pred['discrete'][i][discrete_mask[:,i].bool(),:], 1)
                pred_prob_discrete, pred_class_discrete = torch.max(smax_discrete, 1)
                metrics["val"]["discrete"][var]["ml_confusion_matrix"] += multilabel_confusion_matrix(discrete_gt[:,i][discrete_mask[:,i].bool()].cpu(), pred_class_discrete.cpu(), labels=range(num_clusters['discrete'][i]))
                metrics["val"]["discrete"][var]["confusion_matrix"] += confusion_matrix(discrete_gt[:,i][discrete_mask[:,i].bool()].cpu(), pred_class_discrete.cpu(), labels=range(num_clusters['discrete'][i]))
        

compute_results(metrics, "val", attribute_names, num_samples_val)

################################
# Results of the epoch
################################
result_text = "Res acc:{} mae:{} mse:{} m_acc:{}".format(round(metrics["val"]["binary"]["accuracy"],4), round(metrics["val"]["continuous"]["mae"],4), round(metrics["val"]["continuous"]["mse"],4), round(metrics["val"]["discrete"]["accuracy"],3))
print ("{}".format(result_text), flush=True)

for i,var in enumerate(attribute_names["binary"]):
    print ("val", var, round(metrics["val"]["binary"][var]["accuracy"], 3))

for i,var in enumerate(attribute_names["discrete"]):
    print ("val", var, round(metrics["val"]["discrete"][var]["accuracy"], 3))

for i,var in enumerate(attribute_names["continuous"]):
    print ("val", var, round(metrics["val"]["continuous"][var]["mse"], 3))