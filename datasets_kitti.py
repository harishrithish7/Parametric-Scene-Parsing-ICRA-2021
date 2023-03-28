import sys
import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as transforms
from torchvision import transforms as transforms_m

sys.path.append('../')
class CombinedDataset(Dataset):
    
    def __init__(self, num_clusters, basedir, aerial_range, lookahead_distance, rotate_aerial, aerial_mean, aerial_std, cf_mean=None, cf_std=None, num_ground=5, mode="train", aerial_img_size=256, cf_img_size=(621,188)):
        self.image_type = 'image_02'
        self.num_ground = num_ground
        self.num_previous = self.num_ground -1
        self.aerial_mean = aerial_mean
        self.aerial_std = aerial_std
        self.aerial_img_size = (aerial_img_size, aerial_img_size)
        self.aerial_range = aerial_range
        self.lookahead_distance = lookahead_distance
        self.basedir = basedir
        self.NULL_VALUE = -99999

        self.cf_img_size = cf_img_size
        self.cf_mean = cf_mean
        self.cf_std = cf_std

        self.attributes = {}        
        self.attributes['names'] = {} 
        self.attributes['names']['binary'] = ['mr_is_curve', 'mr_is_oneway', 'mr_has_delimiter','mr_sidewalk_delimiter', 'mr_left_sidewalk', 'mr_right_sidewalk','cw_before_intersect', 'cw_after_intersect', 'cw_left_intersect','cw_right_intersect', 'cw_on_mainroad', 'srl_exists', 'srr_exists','sr_ends_mr']
        self.attributes['names']['discrete'] = ['num_left_lanes', 'num_right_lanes']
        self.attributes['names']['continuous'] = ['mr_rotation', 'srr_width', 'srr_dist', 'srl_width', 'srl_dist', 'mr_delimiter_width', 'cw_on_mainroad_dist', 'mr_sidewalk_delimiter_width', 'mr_curve_radius']

        self.attributes['df'] = {}
        self.attributes['df'] = pd.read_csv('kitti_{}.csv'.format(mode))


        self.transform_ = transforms_m.RandomApply([ transforms_m.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
                    , transforms_m.Grayscale(num_output_channels=3)
                    ], p=0.5)

        if mode == "train":
            self.transform_data =transforms_m.Compose([transforms_m.ToPILImage(), transforms_m.Resize(self.cf_img_size), self.transform_, transforms_m.ToTensor(), transforms_m.Normalize(self.cf_mean, self.cf_std), transforms_m.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False)])
        else:
            self.transform_data =transforms_m.Compose([transforms_m.ToPILImage(), transforms_m.Resize(self.cf_img_size), transforms_m.ToTensor(), transforms_m.Normalize(self.cf_mean, self.cf_std)])

        seqs_list = []
        seqs_meta_path_template = os.path.join(self.basedir,'seqs-{}.txt')
        with open(seqs_meta_path_template.format(mode)) as f:
            data = f.read()
            seqs_list += data.split("\n")[:-1]
        seqs_list = [tuple(line.split(" ")) for line in seqs_list]
        
        self.seqs_meta = {}
        for t in seqs_list:
            seq_full, st, end = t
            seq = seq_full.replace('city_','').replace('residential_','').replace('road_','')
            date = '_'.join(seq.split('_')[:3])
            local_dict = {
                'start_frame': int(st),
                'end_frame': int(end)-1,
                'full_name': seq_full,
                'seq': seq,
                'date': date
            }
            self.seqs_meta[seq] = local_dict

        b_freq = {}
        for var in self.attributes['names']['binary']:
            b_freq[var] = {0: 0., 1: 0.}

        self.cf = {}      
        self.cf_img_paths = [] 
        self.cf2neighbors = {}
        self.cf2wgs = {}
        self.cf2seq = {}
        self.cf2frame = {}
        self.name = "kitti"


        for seq, meta in self.seqs_meta.items():
            for frame in range(meta['start_frame'], meta['end_frame']+1):
                img_path = os.path.join(self.basedir, 'data', self.seqs_meta[seq]['date'], seq, self.image_type, 'data', ('0'*10 + str(frame))[-10:] + '.png')
                
                lat, long = self.get_oxts((seq,frame))
                self.cf2wgs[img_path] = (long, lat)

                if frame >= self.num_previous:
                    self.cf_img_paths.append(img_path)
                    
                    self.cf2seq[img_path] = seq
                    self.cf2frame[img_path] = frame

                    self.cf2neighbors[img_path] = self.get_neighbors(seq, frame)

        # Weights for weighted softmax
        if mode == 'train':
            self.attributes['weightage'] = {"binary": {}, "continuous": None, "discrete": {}}

            b_freq = {}
            for var in self.attributes['names']['binary']:
                vc = self.attributes['df'][var].value_counts()
                b_freq[var] = {}
                b_freq[var][0] = vc[0]
                b_freq[var][1] = vc[1]

            for var in self.attributes['names']['binary']:
                N = b_freq[var][0] + b_freq[var][1]
                self.attributes['weightage']['binary'][var] = {}
                for cls in [0,1]:
                    self.attributes['weightage']['binary'][var][cls] = 1. - b_freq[var][cls]/float(N)

            d_freq = {}
            for i, var in enumerate(self.attributes['names']['discrete']):
                vc = self.attributes['df'][var].value_counts()
                d_freq[var] = {}
                for cls in range(num_clusters['discrete'][i]):
                    d_freq[var][cls] = vc[cls]            

            for i, var in enumerate(self.attributes['names']['discrete']):
                freq_list = []
                for cls in range(num_clusters['discrete'][i]):
                    freq_list.append(d_freq[var][cls])
                lcm = np.lcm.reduce(freq_list)
                weights = lcm/ np.array(freq_list) 
                weights = weights / np.sum(weights)

                self.attributes['weightage']['discrete'][var] = {}
                for cls in range(num_clusters['discrete'][i]):
                    self.attributes['weightage']['discrete'][var][cls] = weights[cls]


        np.random.seed(0)
        np.random.shuffle(self.cf_img_paths)

        if rotate_aerial:
            self.cf2aerial = {cf_path: cf_path.replace('kitti/data', 'kitti/aerial_rot_{}'.format(self.aerial_range)).replace('{}/data'.format(self.image_type),'') for cf_path in self.cf_img_paths}
        else:
            self.cf2aerial = {cf_path: cf_path.replace('kitti/data', 'kitti/aerial_{}'.format(self.aerial_range)).replace('{}/data'.format(self.image_type),'') for cf_path in self.cf_img_paths}

    def get_oxts(self, args):
        seq, frame = args
        oxts_path = os.path.join(self.basedir, 'data', self.seqs_meta[seq]['date'], seq, 'oxts', 'data', ('0'*10 + str(frame))[-10:] + '.txt')
        with open(oxts_path) as f:
            oxts = f.read()
            oxts = oxts.replace('\n','').split(' ')
            lat, long = float(oxts[0]), float(oxts[1])

        return (lat, long)

    def get_neighbors(self, seq, ref_frame):
        neigh_paths = []
        for frame in range(ref_frame-1, ref_frame-self.num_previous-1, -1):
            neigh_path = os.path.join(self.basedir, 'data', self.seqs_meta[seq]['date'], seq, self.image_type, 'data', ('0'*10 + str(frame))[-10:] + '.png')
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

        # load ground location
        cf_wgs = []
        ref_cf_wgs = self.cf2wgs[ref_cf_img_path][::-1]
        cf_wgs.append(ref_cf_wgs)
        for cf_img_path in neighbor_paths:
            cf_wgs.append(self.cf2wgs[cf_img_path][::-1])
        cf_wgs = torch.FloatTensor(cf_wgs)

        # load aerial image
        aerial_img_path = self.cf2aerial[ref_cf_img_path]
        aerial_img = np.array(Image.open(aerial_img_path))[:,:,:3]
        aerial_img = np.transpose(aerial_img, (2, 0, 1))
        aerial_img = torch.from_numpy(aerial_img).float()
        aerial_img /= 255.
        aerial_img =  transforms.normalize(aerial_img, self.aerial_mean, self.aerial_std)
        
        # load aerial location
        aerial_wgs = torch.FloatTensor(ref_cf_wgs)
        
        # load gt
        seq = self.cf2seq[ref_cf_img_path]
        ref_frame = self.cf2frame[ref_cf_img_path]

        record = self.attributes['df'].query("seq == '{}' & frame == {}".format(seq, ref_frame))
        binary_gt = []
        binary_mask = []
        for attr_name in self.attributes['names']['binary']:
            val = record[attr_name].values[0]
            if val == True:
                binary_gt.append(1.0)
                binary_mask.append(1)
            elif val == False:
                binary_gt.append(0.0)
                binary_mask.append(1)
            else:
                binary_gt.append(self.NULL_VALUE)
                binary_mask.append(0)

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
            "seq": seq,
            "ref_frame": ref_frame,
            "ref_wgs": ref_cf_wgs,
            "ref_cf_img_path": ref_cf_img_path
        }

        return aerial_wgs, aerial_img, cf_wgs, cf_imgs, binary_gt, binary_mask, continuous_gt, continuous_mask, discrete_gt, discrete_mask,  meta

    def __len__(self):
        return len(self.cf_img_paths)
