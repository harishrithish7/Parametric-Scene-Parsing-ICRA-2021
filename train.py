import numpy as np
import os
from tqdm import tqdm
import math
import argparse
from datetime import datetime
import logging
import json
import itertools

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from model_combined_branch import CombinedNet
from model_ground_branch import GroundNet
from model_aerial_branch import AerialNet

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

torch.set_printoptions(precision=7)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), help="expirement name")
parser.add_argument("--checkpoint_path", type=str, default=None, help="path to saved checkpoint file")
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
epochs = 10

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

lr = 1e-4
weight_decay = 1e-8
loss_weights = {}
loss_weights['binary'] = 30
loss_weights['continuous'] = 0.01 
loss_weights['discrete'] = 30  

hparams = {
    "inputs": opt.inputs,
    "lookahead": lookahead_distance,
    "aerial_range": aerial_range,
    "rotate_aerial": rotate_aerial,
    "loss_weights_binary": loss_weights["binary"],
    "loss_weights_continuous": loss_weights["continuous"],
    "loss_weights_discrete": loss_weights["discrete"],
    "lr": lr,
    "weight_decay": weight_decay
}
print (hparams, flush = True)

# mean and standard deviation of ground imagery (train set)
cf_mean = torch.Tensor([0.47226224, 0.47520779, 0.51034265])
cf_std = torch.Tensor([0.20337067, 0.21041222, 0.22835514])

# mean and standard deviation of aerial imagery (train set)
aerial_mean = torch.Tensor([0.42290114, 0.41879349, 0.41141361])
aerial_std = torch.Tensor([0.22428898, 0.22103066, 0.21567185])

datadir =  opt.datadir
checkpoint_dir = "checkpoints/"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(os.path.join(checkpoint_dir, opt.exp_name), exist_ok=True)

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
#                        Optimizer Initialization
###########################################################################
# do not update parameters of layers 6-8 of DRN
if opt.inputs == "aerial" or opt.inputs == "combined":
    aerial_features_to_update = [list(module.parameters())  for l in range(6, 9)  for name, module in model.module.arl_features_extraction.named_modules() if "layer{}".format(l) in name and not "layer{}.".format(l) in name]
    aerial_features_to_update = list(itertools.chain.from_iterable(aerial_features_to_update))
    aerial_params = aerial_features_to_update
    aerial_params += model.module.arl_features_extraction.branch_modules.parameters()

if opt.inputs == "sv" or opt.inputs == "combined":
    ground_features_to_update = [list(module.parameters())  for l in range(6, 9)  for name, module in model.module.gnd_features_extraction.named_modules() if "layer{}".format(l) in name and not "layer{}.".format(l) in name]
    ground_features_to_update = list(itertools.chain.from_iterable(ground_features_to_update))
    ground_params = ground_features_to_update
    ground_params += model.module.gnd_features_extraction.branch_modules.parameters()

params = []
if opt.inputs == 'combined':
    mt_weight_params = list(model.module.mt_weights_binary.parameters()) + list(model.module.mt_weights_continuous.parameters()) + list(model.module.mt_weights_discrete.parameters())
params += list(model.module.task_binary.parameters()) + list(model.module.task_continuous.parameters()) + list(model.module.task_discrete.parameters())

if opt.inputs == "combined":
    optimizer = optim.Adam([
        {'params': aerial_params+ground_params+params},
        {'params': mt_weight_params, 'lr':100*lr}], lr=lr, weight_decay=weight_decay)
elif opt.inputs == "sv":
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif opt.inputs == "aerial":
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
###########################################################################
#                Initalize/Load Model & Optimizer state dict
###########################################################################
# Weight initialization from Gaussian Distribution
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

    if classname.find('ConvTranspose') != -1:
        m.bias.data.fill_(0.01)

if opt.inputs == "aerial":
    model.module.arl_features_extraction.branch_modules.apply(weights_init)
    checkpoint_path = opt.checkpoint_path
    if checkpoint_path:
        checkpoint = torch.load(opt.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        score = checkpoint['score']
        best_score = score
    else:
        epoch = 1
        best_score = -math.inf

elif opt.inputs == "sv":
    model.module.gnd_features_extraction.branch_modules.apply(weights_init)
    checkpoint_path = opt.checkpoint_path
    if checkpoint_path:
        checkpoint = torch.load(opt.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        score = checkpoint['score']
        best_score = score
    else:
        epoch = 1
        best_score = -math.inf
else:
    model.module.arl_features_extraction.branch_modules.apply(weights_init)
    model.module.gnd_features_extraction.branch_modules.apply(weights_init)
    if opt.checkpoint_path:
        checkpoint = torch.load(opt.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        score = checkpoint['score']
        best_score = score
    else:
        epoch = 1
        best_score = -math.inf

model.train()
if cuda:
    model.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

###########################################################################
#                           Dataloader
###########################################################################
train_dataset = CombinedDataset(num_clusters, datadir, aerial_range, lookahead_distance, rotate_aerial, aerial_mean, aerial_std, cf_mean, cf_std, num_ground=1, mode="train")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        pin_memory=True)

val_dataset = CombinedDataset(num_clusters, datadir, aerial_range, lookahead_distance, rotate_aerial, aerial_mean, aerial_std, cf_mean, cf_std, num_ground=1, mode="val")
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        pin_memory=True)    

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
    "train": {
        "binary": {},
        "continuous": {},
        "discrete": {}
    }, 
    "val": {
        "binary": {},
        "continuous": {},
        "discrete": {}
    }, 
}

##################################################################################
#                              Loss functions
##################################################################################
attribute_names = train_dataset.attributes['names']
attribute_weights = train_dataset.attributes['weightage']

loss_fn = {"binary": {}, "continuous": None, "discrete": {}}

for var in attribute_names["binary"]:
    weights = attribute_weights["binary"][var]
    weights = torch.Tensor([weights[0], weights[1]]).to(device)
    loss_fn['binary'][var] = nn.CrossEntropyLoss(weight=weights, reduction='sum').cuda()

for i,var in enumerate(attribute_names["discrete"]):
    weights = attribute_weights["discrete"][var]
    weight_list = []
    for cls in range(num_clusters['discrete'][i]):
        weight_list.append(weights[cls])
    weights = torch.Tensor(weight_list).to(device)
    loss_fn['discrete'][var] = nn.CrossEntropyLoss(weight=weights, reduction='sum').cuda()

loss_fn['continuous'] = nn.MSELoss(reduction='sum').cuda()
    
total_batch_num = 1
optimizer.zero_grad()

for epoch in range(epoch,epochs+1):
    ##################################################
    #               Initialize metrics
    ##################################################
    for i, var in enumerate(attribute_names["binary"]):
        metrics["train"]["binary"][var] = {"confusion_matrix": np.zeros((2,2))}
        metrics["val"]["binary"][var] = {"confusion_matrix": np.zeros((2,2))}

    for i, var in enumerate(attribute_names["continuous"]):
        metrics["train"]["continuous"][var] = {"mse": 0., "mae":0.}
        metrics["val"]["continuous"][var] = {"mse": 0., "mae":0.}

    for i, var in enumerate(attribute_names["discrete"]):
        num = num_clusters["discrete"][i]
        for mode in ["train","val"]:
            metrics[mode]["discrete"][var] = {"ml_confusion_matrix": np.zeros((num,2,2)), "confusion_matrix": np.zeros((num,num))}

    ##################################################
    #         Initialize loss
    ##################################################
    loss_individual = {"binary": {}, "continuous": {}, "discrete": {}}
    for var in attribute_names["binary"]:
        loss_individual['binary'][var] = 0

    for var in attribute_names["discrete"]:
        loss_individual['discrete'][var] = 0

    for var in attribute_names["continuous"]:
        loss_individual['continuous'][var] = 0

    num_samples_train = {
        "binary": 0.,
        "continuous": {},
        "discrete": 0.
    }
    for i, var in enumerate(attribute_names["continuous"]):
        num_samples_train["continuous"][var] = 0.
    num_samples_train["continuous"]["total"] = 0.

    train_loss = {"binary": 0., "continuous": 0., "discrete": 0., "total": 0.}

    train_batch = 0
    pbar_train = tqdm(train_loader)

    for aerial_img_loc, aerial_img, sv_locs, sv_img, binary_gt, binary_mask, continuous_gt, continuous_mask, discrete_gt, discrete_mask, meta in pbar_train:
        
        # load data to GPU
        aerial_img_loc, aerial_img, sv_locs, sv_img, binary_gt, binary_mask, continuous_gt, continuous_mask, discrete_gt, discrete_mask = aerial_img_loc.to(device), aerial_img.to(device), sv_locs.to(device), sv_img.to(device), binary_gt.to(device), binary_mask.to(device), continuous_gt.to(device), continuous_mask.to(device), discrete_gt.to(device), discrete_mask.to(device)

        # Forward pass
        if opt.inputs == "combined":
            pred = model(aerial_img, sv_img)
        elif opt.inputs == "aerial":
            pred = model(aerial_img)
        elif opt.inputs == "sv":
            pred = model(sv_img)

        optimizer.zero_grad()

        ##############################
        # binary
        ##############################
        # loss function
        loss_binary = torch.tensor(0.).to(device)
        for i,var in enumerate(attribute_names['binary']):
            if torch.sum(binary_mask[:,i]) != 0:
                loss_individual_ = loss_fn["binary"][var](pred['binary'][i][binary_mask[:,i].bool(),:], binary_gt[:,i][binary_mask[:,i].bool()].long())
                loss_binary += loss_individual_
                loss_individual['binary'][var] +=  loss_individual_.item()

        num_samples_train["binary"] += torch.sum(binary_mask).item()
        train_loss["binary"] += loss_binary.item()

        # metric
        with torch.no_grad():
            for i,var in enumerate(attribute_names["binary"]):
                if torch.sum(binary_mask[:,i]) != 0:
                    smax_binary = nn.functional.softmax(pred['binary'][i])
                    pred_prob_binary, pred_class_binary = torch.max(smax_binary, 1)
                    metrics["train"]["binary"][var]["confusion_matrix"] += confusion_matrix(binary_gt[:,i][binary_mask[:,i].bool()].cpu(), pred_class_binary[binary_mask[:,i].bool()].cpu(), labels=range(2))

        #############################
        # continuous
        #############################
        # loss function
        loss_continuous = torch.tensor(0.).to(device)
        for i, var in enumerate(attribute_names["continuous"]):
            if torch.sum(continuous_mask[:,i]) != 0:                
                loss_individual_ = loss_fn['continuous'](pred['continuous'][i][continuous_mask[:,i].bool(),:], continuous_gt[continuous_mask[:,i].bool(),i].view(-1,1))
                loss_individual['continuous'][var] +=  loss_individual_.item()
                loss_continuous += loss_individual_
                num_samples_train["continuous"][var] += torch.sum(continuous_mask[:,i])

        num_samples_train["continuous"]["total"] += torch.sum(continuous_mask).item()
        train_loss["continuous"] += loss_continuous.item()

        # metric
        with torch.no_grad():
            for i, var in enumerate(attribute_names["continuous"]):
                if torch.sum(continuous_mask[:,i]) != 0:
                    sq_err = torch.sum((pred['continuous'][i][continuous_mask[:,i].bool(),:]-continuous_gt[continuous_mask[:,i].bool(),i].view(-1,1)).pow(2))
                    abs_err = torch.sum(torch.abs(pred['continuous'][i][continuous_mask[:,i].bool(),:]-continuous_gt[continuous_mask[:,i].bool(),i].view(-1,1)))
                    metrics["train"]["continuous"][var]["mse"] += sq_err
                    metrics["train"]["continuous"][var]["mae"] += abs_err

        ##############################
        # discrete
        ##############################
        # loss function
        loss_discrete = torch.tensor(0.).to(device)
        for i,var in enumerate(attribute_names['discrete']):
            if torch.sum(discrete_mask[:,i]) != 0:
                loss_individual_ = loss_fn['discrete'][var](pred['discrete'][i][discrete_mask[:,i].bool(),:], discrete_gt[:,i][discrete_mask[:,i].bool()].long())
                loss_individual['discrete'][var] +=  loss_individual_.item()
                loss_discrete += loss_individual_
                
        num_samples_train["discrete"] += torch.sum(discrete_mask).item()
        train_loss["discrete"] += loss_discrete.item()

        # metric
        with torch.no_grad():
            for i,var in enumerate(attribute_names["discrete"]):
                if torch.sum(discrete_mask[:,i]) != 0:
                    smax_discrete = nn.functional.softmax(pred['discrete'][i][discrete_mask[:,i].bool(),:], 1)
                    pred_prob_discrete, pred_class_discrete = torch.max(smax_discrete, 1)
                    metrics["train"]["discrete"][var]["ml_confusion_matrix"] += multilabel_confusion_matrix(discrete_gt[:,i][discrete_mask[:,i].bool()].cpu(), pred_class_discrete.cpu(), labels=range(num_clusters['discrete'][i]))
                    metrics["train"]["discrete"][var]["confusion_matrix"] += confusion_matrix(discrete_gt[:,i][discrete_mask[:,i].bool()].cpu(), pred_class_discrete.cpu(), labels=range(num_clusters['discrete'][i]))


        ###############################
        # Final loss
        ###############################
        loss = torch.tensor(0.).to(device)
        if torch.sum(binary_mask) != 0:
            loss += (loss_weights['binary']*loss_binary)/torch.sum(binary_mask)
        if torch.sum(continuous_mask) != 0:
            loss += (loss_weights['continuous']*loss_continuous)/torch.sum(continuous_mask)
        if torch.sum(discrete_mask) != 0:
            loss += (loss_weights['discrete']*loss_discrete)/torch.sum(discrete_mask)

        # Backward pass
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm(model.parameters(), 1.1)
        optimizer.step()

        total_batch_num += 1    
        train_batch += 1

        # compute average losses for printing
        if num_samples_train["binary"]:
            mean_bin_loss = (train_loss["binary"]*loss_weights['binary'])/num_samples_train["binary"]
        else:
            mean_bin_loss = 0.
        if num_samples_train["continuous"]["total"]:
            mean_con_loss = (train_loss["continuous"]*loss_weights['continuous'])/num_samples_train["continuous"]["total"]
        else:
            mean_con_loss = 0.
        if num_samples_train['discrete']:
            mean_dis_loss = (train_loss["discrete"]*loss_weights['discrete'])/num_samples_train['discrete']
        else:
            mean_dis_loss = 0.
        mean_tot_loss = mean_bin_loss + mean_con_loss + mean_dis_loss

        pbar_text = "Loss T:{}, B:{} C:{} D:{}".format(round(mean_tot_loss,3), round(mean_bin_loss,3), round(mean_con_loss,3), round(mean_dis_loss,3))
        pbar_train.set_description(pbar_text)

    ################################
    # Average the loss
    ################################
    train_loss["binary"] *= loss_weights['binary']/num_samples_train["binary"]
    train_loss["continuous"] *= loss_weights['continuous']/num_samples_train["continuous"]["total"]
    train_loss["discrete"] *= loss_weights['discrete']/num_samples_train['discrete']
    train_loss["total"] = train_loss["binary"] + train_loss["continuous"] + train_loss['discrete']

    compute_results(metrics, "train", attribute_names, num_samples_train)

    ################################
    # Print results of the epoch
    ################################
    loss_text = "Weighted Train Loss T:{}, B:{} C:{} D:{}".format(round(train_loss["total"],3), round(train_loss["binary"],3), round(train_loss["continuous"],3), round(train_loss["discrete"],3))
    result_text = "Res acc:{} mae:{} mse:{} m_acc:{}".format(round(metrics["train"]["binary"]["accuracy"],3), round(metrics["train"]["continuous"]["mae"],3), round(metrics["train"]["continuous"]["mse"],3), round(metrics["train"]["discrete"]["accuracy"],3))
    print ("epoch:{} {} {}".format(epoch, loss_text, result_text), flush=True)
    print("Individual Losses:", loss_individual, flush=True)

    if opt.inputs == "combined":
        for i,wt in enumerate(list(model.module.mt_weights_binary.parameters())):
            print ("wt", attribute_names['binary'][i], round(torch.sigmoid(wt).item(),3), flush=True)

        for i,wt in enumerate(list(model.module.mt_weights_continuous.parameters())):
            print ("wt", attribute_names['continuous'][i], round(torch.sigmoid(wt).item(),3), flush=True)
        
        for i,wt in enumerate(list(model.module.mt_weights_discrete.parameters())):
            print ("wt", attribute_names['discrete'][i], round(torch.sigmoid(wt).item(),3), flush=True)

    for i,var in enumerate(attribute_names["binary"]):
        print ("train", var, round(metrics["train"]["binary"][var]["accuracy"], 3))

    for i,var in enumerate(attribute_names["discrete"]):
        print ("train", var, round(metrics["train"]["discrete"][var]["accuracy"], 3))

    for i,var in enumerate(attribute_names["continuous"]):
        print ("train", var, round(metrics["train"]["continuous"][var]["mse"], 3))
    
    #######################################################################################################
    #                                           Validation
    #######################################################################################################
    # Initialize loss
    loss_individual = {"binary": {}, "continuous": {}, "discrete": {}}
    for var in attribute_names["binary"]:
        loss_individual['binary'][var] = 0

    for var in attribute_names["discrete"]:
        loss_individual['discrete'][var] = 0

    for var in attribute_names["continuous"]:
        loss_individual['continuous'][var] = 0

    num_samples_val = {
        "binary": 0.,
        "continuous": {},
        "discrete": 0.
    }
    for i, var in enumerate(attribute_names["continuous"]):
        num_samples_val["continuous"][var] = 0.
    num_samples_val["continuous"]["total"] = 0.

    val_batch = 0
    val_loss = {"binary": 0., "continuous": 0., "discrete": 0., "total": 0.}

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
            # loss function
            loss_binary = torch.tensor(0.).to(device)
            for i,var in enumerate(attribute_names['binary']):
                if torch.sum(binary_mask[:,i]) != 0:
                    loss_individual_ = loss_fn["binary"][var](pred['binary'][i][binary_mask[:,i].bool(),:], binary_gt[:,i][binary_mask[:,i].bool()].long())
                    loss_binary += loss_individual_
                    loss_individual['binary'][var] +=  loss_individual_.item()
                    
                    
            num_samples_val["binary"] += torch.sum(binary_mask).item()
            val_loss["binary"] += loss_binary.item()

            for i,var in enumerate(attribute_names["binary"]):
                if torch.sum(binary_mask[:,i]) != 0:
                    smax_binary = nn.functional.softmax(pred['binary'][i])
                    pred_prob_binary, pred_class_binary = torch.max(smax_binary, 1)
                    metrics["val"]["binary"][var]["confusion_matrix"] += confusion_matrix(binary_gt[:,i][binary_mask[:,i].bool()].cpu(), pred_class_binary[binary_mask[:,i].bool()].cpu(), labels=range(2))

            #############################
            # continuous
            #############################
            # loss function
            loss_continuous = torch.tensor(0.).to(device)
            for i, var in enumerate(attribute_names["continuous"]):
                if torch.sum(continuous_mask[:,i]) != 0:
                    loss_individual_ = loss_fn['continuous'](pred['continuous'][i][continuous_mask[:,i].bool(),:], continuous_gt[continuous_mask[:,i].bool(),i].view(-1,1))
                    loss_individual['continuous'][var] +=  loss_individual_.item()
                    loss_continuous += loss_individual_
                    num_samples_val["continuous"][var] += torch.sum(continuous_mask[:,i])
                    
            num_samples_val["continuous"]["total"] += torch.sum(continuous_mask).item()
            val_loss["continuous"] += loss_continuous.item()

            # metric
            for i, var in enumerate(attribute_names["continuous"]):
                if torch.sum(continuous_mask[:,i]) != 0:
                    sq_err = torch.sum((pred['continuous'][i][continuous_mask[:,i].bool(),:]-continuous_gt[continuous_mask[:,i].bool(),i].view(-1,1)).pow(2))
                    abs_err = torch.sum(torch.abs(pred['continuous'][i][continuous_mask[:,i].bool(),:]-continuous_gt[continuous_mask[:,i].bool(),i].view(-1,1)))
                    metrics["val"]["continuous"][var]["mse"] += sq_err
                    metrics["val"]["continuous"][var]["mae"] += abs_err

            ##############################
            # discrete
            ##############################
            # loss function
            loss_discrete = torch.tensor(0.).to(device)
            for i,var in enumerate(attribute_names['discrete']):
                if torch.sum(discrete_mask[:,i]) != 0:
                    loss_individual_ = loss_fn['discrete'][var](pred['discrete'][i][discrete_mask[:,i].bool(),:], discrete_gt[:,i][discrete_mask[:,i].bool()].long())
                    loss_individual['discrete'][var] +=  loss_individual_.item()
                    loss_discrete += loss_individual_
            
            num_samples_val["discrete"] += torch.sum(discrete_mask).item()
            val_loss["discrete"] += loss_discrete.item()

            # metric
            for i,var in enumerate(attribute_names["discrete"]):
                if torch.sum(discrete_mask[:,i]) != 0:
                    smax_discrete = nn.functional.softmax(pred['discrete'][i][discrete_mask[:,i].bool(),:], 1)
                    pred_prob_discrete, pred_class_discrete = torch.max(smax_discrete, 1)
                    metrics["val"]["discrete"][var]["ml_confusion_matrix"] += multilabel_confusion_matrix(discrete_gt[:,i][discrete_mask[:,i].bool()].cpu(), pred_class_discrete.cpu(), labels=range(num_clusters['discrete'][i]))
                    metrics["val"]["discrete"][var]["confusion_matrix"] += confusion_matrix(discrete_gt[:,i][discrete_mask[:,i].bool()].cpu(), pred_class_discrete.cpu(), labels=range(num_clusters['discrete'][i]))
            
            ###############################
            # Final loss
            ###############################
            loss = torch.tensor(0.).to(device)
            if torch.sum(binary_mask) != 0:
                loss += (loss_weights['binary']*loss_binary)/torch.sum(binary_mask)
            if torch.sum(continuous_mask) != 0:
                loss += (loss_weights['continuous']*loss_continuous)/torch.sum(continuous_mask)
            if torch.sum(discrete_mask) != 0:
                loss += (loss_weights['discrete']*loss_discrete)/torch.sum(discrete_mask)
            val_batch += 1

            # compute average losses for printing
            if num_samples_val["binary"]:
                mean_bin_loss = (val_loss["binary"]*loss_weights['binary'])/num_samples_val["binary"]
            else:
                mean_bin_loss = 0.
            if num_samples_val["continuous"]["total"]:
                mean_con_loss = (val_loss["continuous"]*loss_weights['continuous'])/num_samples_val["continuous"]["total"]
            else:
                mean_con_loss = 0.
            if num_samples_val['discrete']:
                mean_dis_loss = (val_loss["discrete"]*loss_weights['discrete'])/num_samples_val['discrete']
            else:
                mean_dis_loss = 0.
            mean_tot_loss = mean_bin_loss + mean_con_loss + mean_dis_loss

            pbar_text = "Weighted Loss T:{}, B:{} C:{} D:{}".format(round(mean_tot_loss,3), round(mean_bin_loss,3), round(mean_con_loss,3), round(mean_dis_loss,3))
            pbar_val.set_description(pbar_text)

    
    ################################
    # Average the loss
    ################################
    val_loss["binary"] *= loss_weights['binary']/num_samples_val["binary"]
    val_loss["continuous"] *= loss_weights['continuous']/num_samples_val["continuous"]["total"]
    val_loss["discrete"] *= loss_weights['discrete']/num_samples_val['discrete']
    val_loss["total"] = val_loss["binary"] + val_loss["continuous"] + val_loss['discrete']

    compute_results(metrics, "val", attribute_names, num_samples_val)

    ################################
    # Results of the epoch
    ################################
    loss_text = "Val Loss T:{}, B:{} C:{} D:{}".format(round(val_loss["total"],3), round(val_loss["binary"],3), round(val_loss["continuous"],3), round(val_loss["discrete"],3))
    result_text = "Res acc:{} mae:{} mse:{} m_acc:{}".format(round(metrics["val"]["binary"]["accuracy"],3), round(metrics["val"]["continuous"]["mae"],3), round(metrics["val"]["continuous"]["mse"],3), round(metrics["val"]["discrete"]["accuracy"],3))
    print ("epoch:{} {} {}".format(epoch, loss_text, result_text), flush=True)
    print("Individual Losses:", loss_individual, flush=True)

    for i,var in enumerate(attribute_names["binary"]):
        print ("val", var, round(metrics["val"]["binary"][var]["accuracy"], 3))

    for i,var in enumerate(attribute_names["discrete"]):
        print ("val", var, round(metrics["val"]["discrete"][var]["accuracy"], 3))

    for i,var in enumerate(attribute_names["continuous"]):
        print ("val", var, round(metrics["val"]["continuous"][var]["mse"], 3))


    ################################
    # Save the models
    ################################
    # save best model
    if opt.lookahead == 30:
        score = (metrics["val"]["binary"]["accuracy"]*100-100) + (metrics["val"]["discrete"]["accuracy"]*100-100) + (-metrics["val"]["continuous"]["mse"])
    else:
        score = (metrics["val"]["binary"]["accuracy"]*100-100) + (metrics["val"]["discrete"]["accuracy"]*100-100) + (-metrics["val"]["continuous"]["mae"])
    print ("Score:", score, flush=True)
    if score > best_score: 
        path = os.path.join(checkpoint_dir, opt.exp_name, "best.tar")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score
                }, path)
        best_score = score

    # save last model
    path = os.path.join(checkpoint_dir, opt.exp_name, "last.tar")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score
            }, path)

