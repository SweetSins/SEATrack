# README

## Introduction
- The code for our SEATrack


## Install the environment
**Option1**: Use the Anaconda (CUDA 10.2)
```
conda create -n seatrack python=3.6
conda activate seatrack
bash install.sh
```

**Option2**: Use the Anaconda (CUDA 11.3)
```
conda env create -f seatrack_cuda113_env.yaml
```

**Option3**: Use the docker file

We provide the full docker file here.


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python tracking/train.py --script seatrack --config vitb_mae_ea_32x4_ep350 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/seatrack`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## How to use:
- **Path**: Replace `_C.DATA.ROOT` and `_C.OUTPUT` in `configs/default_img.py` & `default_vid.py` with your own data path and output path, respectively.
- **Training**: For `dataset_name` dataset: 
  ```bash
  python -m torch.distributed.launch --nproc_per_node=2 --master_port 77777 main.py --dataset dataset_name --cfg configs/res50_cels_cal.yaml --gpu 0,1 --spr 0 --sacr 0.05 --rr 1.0
## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX4070Ti GPU.

```
# Profiling vitb_mae_ea_32x4_ep350
python tracking/profile_model.py --script seatrack --config vitb_mae_ea_32x4_ep350
# Profiling vitb_Large_mae_ea_32x4_ep350
python tracking/profile_model.py --script seatrack --config vitb_Large_mae_ea_32x4_ep350
```
