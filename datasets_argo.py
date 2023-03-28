import os
import sys
from PIL import Image
import numpy as np
import pandas as pd
import itertools

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from classes import Ground, DAMap

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as transforms
from torchvision import transforms as transforms_m

sys.path.append('../')

class CombinedDataset(Dataset):
    
    def __init__(self, num_clusters, tracking_dir, aerial_range, lookahead_distance, rotate_aerial, aerial_mean, aerial_std, cf_mean=None, cf_std=None, num_ground=5, mode="train", aerial_img_size=256, cf_img_size = (400,640)):
        self.camera = 'ring_front_center'
        self.num_ground = num_ground
        self.num_previous = self.num_ground -1
        self.aerial_mean = aerial_mean
        self.aerial_std = aerial_std
        self.aerial_img_size = (aerial_img_size, aerial_img_size)
        self.aerial_range = aerial_range
        self.lookahead_distance = lookahead_distance
        self.cf_mean = cf_mean
        self.cf_std = cf_std
        self.NULL_VALUE = -99999

        self.argoverse_loader = ArgoverseTrackingLoader(os.path.join(tracking_dir, mode))

        self.cf_img_size = cf_img_size
        
        self.da_map = {}
        self.da_map['MIA'] = DAMap('MIA')
        self.da_map['PIT'] = DAMap('PIT')


        self.attributes = {}        
        self.attributes['names'] = {} 

        self.attributes['names']['binary'] = ['lanes_to_left', 'lanes_to_right', 'one_way', 'is_left_side_road', 'is_right_side_road', 'at_intersection', 'main_road_continues_1', 'is_road_curved']
        self.attributes['names']['discrete'] = ['num_lanes_to_left', 'num_lanes_to_right']
        self.attributes['names']['continuous'] = ['distance_to_intersection_1', 'curvature_radius']

        self.attributes['df'] = {}
        self.attributes['df']['PIT'] = pd.read_csv('Argoverse_preprocessed_data/attribute_data_PIT_{}_{}_pp.csv'.format(mode, self.lookahead_distance))
        self.attributes['df']['MIA'] = pd.read_csv('Argoverse_preprocessed_data/attribute_data_MIA_{}_{}_pp.csv'.format(mode, self.lookahead_distance))

        
        b_freq = {}
        for var in self.attributes['names']['binary']:
            b_freq[var] = {0: 0., 1: 0.}

        d_freq = {}
        for i, var in enumerate(self.attributes['names']['discrete']):
            d_freq[var] = {}
            for cls in range(num_clusters['discrete'][i]):
                d_freq[var][cls] = 0.

        self.cf = {}      
        self.cf_img_paths = [] 
        self.cf2neighbors = {}
        self.cf2wgs = {}
        self.cf2region = {}
        self.cf2translations = {}

        self.transform_ = transforms_m.RandomApply([ transforms_m.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
                    , transforms_m.Grayscale(num_output_channels=3)
                    ], p=0.5)

        if mode == "train":
            self.transform_data =transforms_m.Compose([transforms_m.ToPILImage(), transforms_m.Resize(self.cf_img_size), self.transform_, transforms_m.ToTensor(), transforms_m.Normalize(self.cf_mean, self.cf_std), transforms_m.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)])
        else:
            self.transform_data =transforms_m.Compose([transforms_m.ToPILImage(), transforms_m.Resize(self.cf_img_size), transforms_m.ToTensor(), transforms_m.Normalize(self.cf_mean, self.cf_std)])


        for region in ['MIA','PIT']:
            self.cf[region] = Ground([self.argoverse_loader], region, self.camera)
            
            paths_with_neighbors = list(filter(lambda x: self.cf[region].path2sample[x] >= self.num_previous, self.cf[region].img_paths))
            neighbors_map_obj = map(self.get_neighbors, paths_with_neighbors, itertools.repeat(region, len(paths_with_neighbors)))
            path2neighbors = {ref_path:neighbors for ref_path,neighbors in zip(paths_with_neighbors,neighbors_map_obj)}

            self.cf_img_paths.extend(paths_with_neighbors)
            self.cf2neighbors = {**self.cf2neighbors, **path2neighbors}

            self.cf2wgs = {**self.cf2wgs, **self.cf[region].path2wgs}

            region_meta = {path: region for path in self.cf[region].img_paths}
            self.cf2region = {**self.cf2region, **region_meta}

            self.cf2translations = {**self.cf2translations, **self.cf[region].path2translations}

            for var in self.attributes['names']['binary']:
                vc = self.attributes['df'][region][var].value_counts()
                b_freq[var][0] += vc[0]
                b_freq[var][1] += vc[1]

            for i, var in enumerate(self.attributes['names']['discrete']):
                vc = self.attributes['df'][region][var].value_counts()
                for cls in range(num_clusters['discrete'][i]):
                    if cls in vc.keys():
                        d_freq[var][cls] += vc[cls]      

                
        self.attributes['weightage'] = {"binary": {}, "continuous": None, "discrete": {}}
        for var in self.attributes['names']['binary']:
            N = b_freq[var][0] + b_freq[var][1]
            self.attributes['weightage']['binary'][var] = {}
            for cls in [0,1]:
                self.attributes['weightage']['binary'][var][cls] = 1. - b_freq[var][cls]/float(N)

        for i, var in enumerate(self.attributes['names']['discrete']):
            freq_list = []
            for cls in range(num_clusters['discrete'][i]):
                freq_list.append(int(d_freq[var][cls]))
            lcm = np.lcm.reduce(freq_list)
            weights = lcm/ np.array(freq_list) 
            weights = weights / np.sum(weights)

            self.attributes['weightage']['discrete'][var] = {}
            for cls in range(num_clusters['discrete'][i]):
                self.attributes['weightage']['discrete'][var][cls] = weights[cls]


        np.random.seed(0)
        np.random.shuffle(self.cf_img_paths)

        if rotate_aerial:
            self.cf2aerial = {cf_path: cf_path.replace('argoverse-tracking','aerial_rot_{}'.format(self.aerial_range)).replace('.png','.jpg') for cf_path in self.cf_img_paths}
        else:
            self.cf2aerial = {cf_path: cf_path.replace('argoverse-tracking','aerial').replace('.png','.jpg') for cf_path in self.cf_img_paths}

        
    def get_neighbors(self, ref_path, region):
        log = self.cf[region].path2log[ref_path]
        ref_sample = self.cf[region].path2sample[ref_path]

        neigh_paths = []
        for sample in range(ref_sample-1, ref_sample-self.num_previous-1, -1):
            neigh_path = self.argoverse_loader.get(log).get_image_list_sync(self.camera)[sample]
            neigh_paths.append(neigh_path)

        return neigh_paths

    def __getitem__(self, index):
        # load ground images
        ref_cf_img_path = self.cf_img_paths[index]
        neighbor_paths = self.cf2neighbors[ref_cf_img_path]
        cf_imgs = torch.zeros((self.num_ground, 3, *self.cf_img_size))
        IM = transforms.to_tensor(Image.open(ref_cf_img_path))#.permute(1,2,0)
        ref_cf_img = self.transform_data(IM)
        cf_imgs[0] = ref_cf_img
        for i, cf_img_path in enumerate(neighbor_paths):
            IM = transforms.to_tensor(Image.open(cf_img_path)) #.permute(1,2,0)
            cf_img = self.transform_data(IM)
            cf_imgs[i+1] = cf_img

        # load ground img. location
        cf_wgs = []        
        ref_cf_wgs = self.cf2wgs[ref_cf_img_path][::-1]
        cf_wgs.append(ref_cf_wgs)
        for cf_img_path in neighbor_paths:
            cf_wgs.append(self.cf2wgs[cf_img_path][::-1])
        cf_wgs = torch.FloatTensor(cf_wgs)

        # load aerial image
        aerial_img_path = self.cf2aerial[ref_cf_img_path]
        aerial_img = np.array(Image.open(aerial_img_path))
        aerial_img = np.transpose(aerial_img, (2, 0, 1))
        aerial_img = torch.from_numpy(aerial_img).float()
        aerial_img /= 255.
        aerial_img =  transforms.normalize(aerial_img, self.aerial_mean, self.aerial_std)
        
        # load aerial location
        aerial_wgs = torch.FloatTensor(ref_cf_wgs)
        
        # load gt
        region = self.cf2region[ref_cf_img_path]
        log = self.cf[region].path2log[ref_cf_img_path]
        ref_sample = self.cf[region].path2sample[ref_cf_img_path]

        record = self.attributes['df'][region].query("log_index == '{}' & frame_index == {}".format(log, ref_sample))
        binary_gt = []
        binary_mask = []
        for attr_name in self.attributes['names']['binary']:
            val = record[attr_name].values[0]
            if val == 'True':
                binary_gt.append(1.0)
                binary_mask.append(1)
            elif val == 'False':
                binary_gt.append(0.0)
                binary_mask.append(1)
            else:
                val = float(val)
                binary_gt.append(val)
                if val == self.NULL_VALUE:
                    binary_mask.append(0)
                else:
                    binary_mask.append(1)
        binary_gt = torch.FloatTensor(binary_gt)
        binary_mask = torch.FloatTensor(binary_mask)

        continuous_gt = []
        continuous_mask = []
        for attr_name in self.attributes['names']['continuous']:
            val = float(record[attr_name])
            continuous_gt.append(val)
            if val == self.NULL_VALUE:
                continuous_mask.append(0)
            else:
                continuous_mask.append(1)
        continuous_gt = torch.FloatTensor(continuous_gt)
        continuous_mask = torch.FloatTensor(continuous_mask)

        discrete_gt = []
        discrete_mask = []
        for attr_name in self.attributes['names']['discrete']:
            val = float(record[attr_name])
            discrete_gt.append(val)
            if val == self.NULL_VALUE:
                discrete_mask.append(0)
            else:
                discrete_mask.append(1)
        discrete_gt = torch.FloatTensor(discrete_gt)
        discrete_mask = torch.FloatTensor(discrete_mask)

        meta = {
            "region": region,
            "log": log,
            "ref_sample": ref_sample,
            "ref_wgs": ref_cf_wgs,
            "ref_cf_img_path": ref_cf_img_path
        }
        return aerial_wgs, aerial_img, cf_wgs, cf_imgs, binary_gt, binary_mask, continuous_gt, continuous_mask, discrete_gt, discrete_mask,  meta

    def __len__(self):
        return len(self.cf_img_paths)
