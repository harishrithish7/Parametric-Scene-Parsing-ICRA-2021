Code for the paper, 'Looking Farther in Parametric Scene Parsing with Ground and Aerial Imagery', ICRA 2021.

Link to the dataset: https://zenodo.org/record/4091397

## Data Download

Download Argoverse 3D Tracking training and validation sets from https://www.argoverse.org/data.html#download-link

Download Argoverse HD map files from https://www.argoverse.org/data.html#download-link and store in root directory

Download KITTI dataset from http://www.cvlibs.net/datasets/kitti and annotations from http://www.nec-labs.com/~mas/BEV/

Download Aerial images for KITTI and Argoverse from https://drive.google.com/drive/folders/1irJkc5PFbmcEZxfgV6wqXSzM9SdlxonU?usp=sharing

Your data structure would now look like

argoverse<br/>
|___aerial_rot_40<br/>
&emsp;|___train<br/>
&emsp;&emsp;|&emsp;02cf0ce1-699a-373b-86c0-eb6fd5f4697a<br/>
&emsp;&emsp;|&emsp;...<br/>
&emsp;|___val<br/>
&emsp;&emsp;|&emsp;00c561b9-2057-358d-82c6-5b06d76cebcf<br/>
&emsp;&emsp;|&emsp;...<br/>
|___aerial_rot_70<br/>
&emsp;|___train<br/>
&emsp;&emsp;|&emsp;02cf0ce1-699a-373b-86c0-eb6fd5f4697a<br/>
&emsp;&emsp;|&emsp;...<br/>
&emsp;|___val<br/>
&emsp;&emsp;|&emsp;00c561b9-2057-358d-82c6-5b06d76cebcf<br/>
&emsp;&emsp;|&emsp;...<br/>
|___argoverse-tracking<br/>
&emsp;|___train<br/>
&emsp;&emsp;|&emsp;02cf0ce1-699a-373b-86c0-eb6fd5f4697a<br/>
&emsp;&emsp;|&emsp;...<br/>
&emsp;|___val<br/>
&emsp;&emsp;|&emsp;00c561b9-2057-358d-82c6-5b06d76cebcf<br/>
&emsp;&emsp;|&emsp;...<br/>
kitti<br/>
|___aerial_rot_40<br/>
&emsp;|___2011_09_26<br/>
&emsp;&emsp;|&emsp;2011_09_26_drive_0001_sync<br/>
&emsp;&emsp;|&emsp;...<br/>
&emsp;|&emsp;...<br/>
|___aerial_rot_70<br/>
&emsp;|___2011_09_26<br/>
&emsp;&emsp;|&emsp;2011_09_26_drive_0001_sync<br/>
&emsp;&emsp;|&emsp;...<br/>
&emsp;|&emsp;...<br/>
|___data<br/>
&emsp;|___2011_09_26<br/>
&emsp;&emsp;|&emsp;2011_09_26_drive_0001_sync<br/>
&emsp;&emsp;|&emsp;...<br/>
&emsp;|&emsp;...<br/>
|___gt<br/>
&emsp;|___city_2011_09_26_drive_0001_sync<br/>
&emsp;|&emsp;...<br/>
|&emsp;seqs-train.txt<br/>
|&emsp;seqs-val.txt<br/>


## Training
python train.py --lookahead <> --inputs <> --datadir <> --dataset <>

## Evaluation
python eval.py --lookahead <> --inputs <> --datadir <> --checkpoint_path <> --dataset <>

lookahead: 30 or 60<br/>
inputs: aerial or sv or combined<br/>
dataset: argo or kitti<br/>
datadir: path to data directory (.../argoverse-tracking or .../kitti)<br/>
checkpoint_path: path to trained model


