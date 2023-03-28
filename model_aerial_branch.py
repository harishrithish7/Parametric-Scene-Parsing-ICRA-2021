#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F
from drn import drn
import copy

###############################################################################
#       Identity function to remove classification layers of DRN
###############################################################################
class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
	def forward(self, x):
		return x

###############################################################################
#               Feature extraction from Aerial Imagery using DRNs
###############################################################################
class Aerial_DRN(nn.Module):
	def __init__(self, num_classes, branch_layer, pretrained=True):
		super(Aerial_DRN, self).__init__()
		self.Aerial_feature_extractor = drn.drn_d_105(pretrained=pretrained)
		self.Aerial_feature_extractor.avgpool = Identity()
		self.Aerial_feature_extractor.fc = Identity()

		self.branch_layer = branch_layer
		self.branch_modules = nn.ModuleDict()
		self.num_classes = num_classes
		
		new_layers = []
		add_layers = False

		for layer_name, module in self.Aerial_feature_extractor._modules.items():
			if (add_layers or layer_name == branch_layer) and 'layer' in layer_name:
				add_layers = True
				new_layers.append(module)

		for attr_type in ['binary','continuous','discrete']:
			self.branch_modules[attr_type] = nn.ModuleList([nn.Sequential(*copy.deepcopy(new_layers)) for _ in range(num_classes[attr_type])])	

	def forward(self, x):
		for layer_name, module in self.Aerial_feature_extractor._modules.items():
			if layer_name == self.branch_layer:
				break
			x = module(x)

		op = {}
		for attr_type in ['binary','continuous','discrete']:
			op[attr_type] = []
			for i in range(self.num_classes[attr_type]):
				branch_op = self.branch_modules[attr_type][i](x)
				op[attr_type].append(branch_op)

		return op

###############################################################################
#                             Remote model
###############################################################################
class  AerialNet(nn.Module):
	def __init__(self, num_classes, num_clusters, branch_layer, dataset):
		super(AerialNet, self).__init__()
		self.num_clusters = num_clusters
		self.num_classes = num_classes

		self.arl_features_extraction = Aerial_DRN(num_classes, branch_layer)

		if dataset == "argo":
			self.adaptive_roi_pool = nn.AdaptiveMaxPool2d((28,28))
		elif dataset == "kitti":
			self.adaptive_roi_pool = nn.AdaptiveMaxPool2d((14,14))
		
		self.task_binary = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 512, 1), nn.ReLU(), nn.Conv2d(512, 256, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1)), nn.Flatten(), nn.Linear(256, 2)) for _ in range(num_classes['binary'])])
		
		self.task_continuous = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 512, 1), nn.ReLU(), nn.Conv2d(512, 256, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1)), nn.Flatten(), nn.Linear(256, 1)) for _ in range(num_classes['continuous'])])

		self.task_discrete = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 512, 1), nn.ReLU(), nn.Conv2d(512, 256, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1)), nn.Flatten(), nn.Linear(256, num)) for num in num_clusters['discrete']])
	
	def forward(self, aerial_image):
		aerial_features = self.arl_features_extraction(aerial_image)
		
		out_binary = []
		out_continuous = []
		out_discrete = []
		for attr_type in ["binary", "discrete", "continuous"]:
			for i in range(self.num_classes[attr_type]):
				aerial_pool_features = self.adaptive_roi_pool(aerial_features[attr_type][i])
				if attr_type == 'binary':
					out_binary.append(self.task_binary[i](aerial_pool_features))
				elif attr_type == 'continuous':
					out_continuous.append(self.task_continuous[i](aerial_pool_features))
				else:
					out_discrete.append(self.task_discrete[i](aerial_pool_features))

		return {'binary': out_binary, 'continuous': out_continuous, 'discrete': out_discrete}

