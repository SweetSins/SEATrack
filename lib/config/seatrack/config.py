from easydict import EasyDict as edict
import yaml

"""
Add default config for SEATrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.DROP_PATH_RATE = 0.1
cfg.TRAIN.ELASTIC_ATTN_STATE = 0
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.EPOCH = 350
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.LR = 0.0004
cfg.TRAIN.LR_DROP_EPOCH = 240
cfg.TRAIN.NUM_WORKER = 10
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.AMP = False

# DATA
cfg.DATA = edict()
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.CENTER_JITTER = 3
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.SCALE_JITTER = 0.25
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain", "COCO17", "TRACKINGNET"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1, 1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000

# TEST
cfg.TEST = edict()
cfg.TEST.EPOCH = 300
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128

def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return

def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError(f"{k} not exist in config.py")
    else:
        return

def update_config_from_file(filename, base_cfg=None):
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)