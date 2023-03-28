#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F
from drn import drn
import copy

###############################################################################
# identity function to remove the layers in the network
###############################################################################
class Identity(nn.Module):
     def __init__(self):
          super(Identity, self).__init__()
     def forward(self, x):
          return x

###############################################################################
# Feature extraction from Ground Imagery using DRNs
###############################################################################

class Ground_DRN(nn.Module):
	def __init__(self, num_classes, branch_layer, pretrained=True):
		super(Ground_DRN, self).__init__()
		self.Ground_feature_extractor = drn.drn_d_105(pretrained=pretrained)
		self.Ground_feature_extractor.avgpool = Identity()
		self.Ground_feature_extractor.fc = Identity()

		self.branch_layer = branch_layer
		self.branch_modules = nn.ModuleDict()
		self.num_classes = num_classes

		new_layers = []
		add_layers = False

		for layer_name, module in self.Ground_feature_extractor._modules.items():
			if (add_layers or layer_name == branch_layer) and 'layer' in layer_name:
				add_layers = True
				new_layers.append(module)

		for attr_type in ['binary','continuous','discrete']:
			self.branch_modules[attr_type] = nn.ModuleList([nn.Sequential(*copy.deepcopy(new_layers)) for _ in range(num_classes[attr_type])])	

	def forward(self, x):
		x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
		
		for layer_name, module in self.Ground_feature_extractor._modules.items():
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
#                           Proximate Model
###############################################################################
class  GroundNet(nn.Module):
	def __init__(self, num_classes, num_clusters, branch_layer, dataset):
		super(GroundNet, self).__init__()
		self.num_clusters = num_clusters
		self.num_classes = num_classes
		
		self.gnd_features_extraction = Ground_DRN(num_classes, branch_layer)

		if dataset == "argo":
			self.adaptive_roi_pool = nn.AdaptiveMaxPool2d((28,28))
		elif dataset == "kitti":
			self.adaptive_roi_pool = nn.AdaptiveMaxPool2d((14,14))

		self.task_binary = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 512, 1), nn.ReLU(), nn.Conv2d(512, 256, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1)), nn.Flatten(), nn.Linear(256, 2)) for _ in range(num_classes['binary'])])	  
		
		self.task_continuous = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 512, 1), nn.ReLU(), nn.Conv2d(512, 256, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1)), nn.Flatten(), nn.Linear(256, 1)) for _ in range(num_classes['continuous'])])

		self.task_discrete = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 512, 1), nn.ReLU(), nn.Conv2d(512, 256, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1)), nn.Flatten(), nn.Linear(256, num)) for num in num_clusters['discrete']])	  

	
	def forward(self, ground_image):
		
		ground_features = self.gnd_features_extraction(ground_image)
		
		out_binary = []
		out_continuous = []
		out_discrete = []
		for attr_type in ["binary", "discrete", "continuous"]:
			for i in range(self.num_classes[attr_type]):
				ground_pool_features = self.adaptive_roi_pool(ground_features[attr_type][i])
				if attr_type == 'binary':
					out_binary.append(self.task_binary[i](ground_pool_features))
				elif attr_type == 'continuous':
					out_continuous.append(self.task_continuous[i](ground_pool_features))
				else:
					out_discrete.append(self.task_discrete[i](ground_pool_features))

		return {'binary': out_binary, 'continuous': out_continuous, 'discrete': out_discrete}