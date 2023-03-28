from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import os
from pyproj import Proj
import numpy as np
from glob import glob
import itertools
import matplotlib.pyplot as plt
import logging
import cv2


class DAMap:
    def __init__(self, region):
        if region == 'MIA':
            self.map = np.load('map_files/MIA_10316_driveable_area_mat_2019_05_28.npy')[::-1,:]
            self.sw_disp = [-502.0, 545.0]
            self.origin_utm = [580560.0088, 2850959.9999] # zone 17
            self.ref_proj = Proj("+proj=utm +zone=17T, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
          
        elif region == 'PIT':
            self.map = np.load('map_files/PIT_10314_driveable_area_mat_2019_05_28.npy')[::-1,:]
            self.sw_disp = [642.0, -211.0]
            self.origin_utm = [583710.0070, 4477259.9999] # zone 17
            self.ref_proj = Proj("+proj=utm +zone=17T, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
          
class Ground:
    def __init__(self, argo_loader, region, camera):
        self.region = region
        self.camera = camera
        self.img_paths = []
        
        da_map = DAMap(region)
        self.origin_utm = da_map.origin_utm
        self.ref_proj = da_map.ref_proj
        
        self.img_paths = []
        self.path2utm = {}
        self.path2wgs = {}
        self.path2log = {}
        self.path2sample = {}
        self.path2translations = {}

        for loader in argo_loader:
            for log in loader.log_list:
                argoverse_data = loader.get(log)
                
                if argoverse_data.city_name != self.region:
                    continue
                    
                img_paths = argoverse_data.get_image_list_sync(self.camera)
                self.img_paths.extend(img_paths)

                log_meta = {img_path: log for img_path in img_paths}
                self.path2log = {**self.path2log, **log_meta}

                sample_meta = {img_path: sample for sample,img_path in enumerate(img_paths)}
                self.path2sample = {**self.path2sample, **sample_meta}

                num_samples = len(img_paths)

                translations = list(map(self.get_translation, itertools.repeat(argoverse_data, num_samples), range(num_samples)))
                translations_meta = {img_path: translation for img_path,translation in zip(img_paths,translations)}
                self.path2translations = {**self.path2translations, **translations_meta}

                utm_map_obj = map(self.get_utm, translations)
                utm_meta = {img_path:utm for img_path,utm in zip(img_paths, utm_map_obj)}
                self.path2utm = {**self.path2utm, **utm_meta}
                
                wgs_map_obj = map(self.get_wgs, translations)
                wgs_meta = {img_path:wgs for img_path,wgs in zip(img_paths,wgs_map_obj)}
                self.path2wgs = {**self.path2wgs, **wgs_meta}

    def get_wgs(self, translation):
        pt_utm = [self.origin_utm[0]+translation[0], self.origin_utm[1]+translation[1]]
        pt_wgs = self.ref_proj(*pt_utm, inverse=True)
        return pt_wgs

    def get_utm(self, translation):
        pt_utm = [self.origin_utm[0]+translation[0], self.origin_utm[1]+translation[1]]
        return pt_utm

    def get_translation(self, argoverse_data, sample):
        return argoverse_data.get_pose(sample).translation