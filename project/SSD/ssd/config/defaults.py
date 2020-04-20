from yacs.config import CfgNode as CN

cfg = CN()

cfg.MODEL = CN()
cfg.MODEL.META_ARCHITECTURE = 'SSDDetector'
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
cfg.MODEL.THRESHOLD = 0.5
cfg.MODEL.NUM_CLASSES = 21
# Hard negative mining
cfg.MODEL.NEG_POS_RATIO = 3
cfg.MODEL.CENTER_VARIANCE = 0.1
cfg.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
cfg.MODEL.BACKBONE = CN()
cfg.MODEL.BACKBONE.NAME = 'vgg'
cfg.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 512, 256, 256, 256)
cfg.MODEL.BACKBONE.PRETRAINED = True
cfg.MODEL.BACKBONE.INPUT_CHANNELS = 3

# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------
cfg.MODEL.PRIORS = CN()
# cfg.MODEL.PRIORS.FEATURE_MAPS = [[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]] #[300x300] resnet instructor v2
# cfg.MODEL.PRIORS.STRIDES = [[8,8], [16, 16], [30, 30], [60, 60], [100, 100], [300, 300]] #[300x300] resnet instructor v2
# cfg.MODEL.PRIORS.FEATURE_MAPS = [[40, 30], [20, 15], [10, 8], [5, 5], [3, 3], [1, 1]] #[320x240] resnet instructor, resnet152, googlenet
# cfg.MODEL.PRIORS.STRIDES = [[8, 8], [16, 16], [32, 30], [64, 48], [107, 80], [320, 240]]  #[320x240] resnet instructor, resnet152, googlenet
# cfg.MODEL.PRIORS.FEATURE_MAPS = [[60, 45], [30, 23], [15, 12], [7, 7], [4, 4], [1, 1]] #[480x320] resnet instructor, resnet152, googlenet
# cfg.MODEL.PRIORS.STRIDES = [[8, 8], [16, 15.65], [32, 30], [68.57, 51.43], [120, 90], [480, 360]]  #[480x320] resnet instructor, resnet152, googlenet
# cfg.MODEL.PRIORS.FEATURE_MAPS = [[90, 68],  [45, 34],       [23, 17],       [11, 9],        [6, 5],     [3, 3],         [1, 1]] #[720x560] resnet instructor, resnet152, googlenet
# cfg.MODEL.PRIORS.STRIDES = [[8, 8.23],      [16, 16.47],    [31.3, 32.94],  [65.45, 62.22], [120, 112], [240, 186.66],  [720,560]] #[720x560] resnet instructor, resnet152, googlenet
cfg.MODEL.PRIORS.FEATURE_MAPS = [[135, 102],  [68, 51],       [34, 26],       [17, 14],        [9, 7],     [6,5], [3, 3],         [1, 1]] #[1080x810] resnet instructor, resnet152, googlenet
cfg.MODEL.PRIORS.STRIDES = [[8, 7.94],      [15.88, 15.88],    [31.76, 31.15],  [63.53, 57.86], [120, 115.71], [180, 162], [360, 270],  [1080, 810]] #[1080x810] resnet instructor, resnet152, googlenet

cfg.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264] # [300x300], [320x240], [480x320]
cfg.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315] # [300x300], [320x240], [480x320]
# cfg.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264, 315] # [720x560] resnet instructor small boxes
# cfg.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315, 500] # [720x560] resnet instructor small boxes


# cfg.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]] #[300x300], [320x240], [480x320]
# cfg.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]] #[720x560]
cfg.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]] #[1080x810]
# When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
# #boxes = 2 + #ratio * 2
# cfg.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location [300x300], [320x240], [480x320]
# cfg.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 6, 4, 4]  # number of boxes per feature map location [720, 560] resnet instructor
cfg.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 6, 6, 4, 4]  # number of boxes per feature map location [1080x810] resnet instructor
cfg.MODEL.PRIORS.CLIP = True

# -----------------------------------------------------------------------------
# Box Head
# -----------------------------------------------------------------------------
cfg.MODEL.BOX_HEAD = CN()
cfg.MODEL.BOX_HEAD.NAME = 'SSDBoxHead'
cfg.MODEL.BOX_HEAD.PREDICTOR = 'SSDBoxPredictor'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
# Image size
cfg.INPUT.IMAGE_SIZE = [320, 240]
# Values to be used for image normalization, RGB layout
cfg.INPUT.PIXEL_MEAN = [123, 117, 104]
cfg.INPUT.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
cfg.DATASETS = CN()
# List of the dataset names for training, as present in pathscfgatalog.py
cfg.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in pathscfgatalog.py
cfg.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
cfg.DATA_LOADER = CN()
# Number of data loading threads
cfg.DATA_LOADER.NUM_WORKERS = 16
cfg.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver - The same as optimizer
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
# train configs
cfg.SOLVER.MAX_ITER = 120000
cfg.SOLVER.LR_STEPS = [80000, 100000]
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.BATCH_SIZE = 32
cfg.SOLVER.LR = 5e-3
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 5e-4
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
cfg.SOLVER.WARMUP_ITERS = 500

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
cfg.TEST = CN()
cfg.TEST.NMS_THRESHOLD = 0.45
cfg.TEST.CONFIDENCE_THRESHOLD = 0.01
cfg.TEST.MAX_PER_CLASS = -1
cfg.TEST.MAX_PER_IMAGE = 100
cfg.TEST.BATCH_SIZE = 10

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
cfg.EVAL_STEP = 4000 # Evaluate dataset every eval_step, disabled when eval_step < 0
cfg.MODEL_SAVE_STEP = 15000 # Save checkpoint every save_step
cfg.LOG_STEP = 10 # Print logs every log_stepPrint logs every log_step
cfg.OUTPUT_DIR = "outputs"
cfg.DATASET_DIR = "datasets"