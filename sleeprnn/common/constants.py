"""constants.py: Module that stores several useful constants for the project."""

# Dataset name
MASS_SS_NAME = 'mass_ss'
INTA_SS_NAME = 'inta_ss'
MASS_KC_NAME = 'mass_kc'

# Database split
TRAIN_SUBSET = 'train'
VAL_SUBSET = 'val'
TEST_SUBSET = 'test'
ALL_TRAIN_SUBSET = 'all_train'

# Task mode
N2_RECORD = 'n2'
WN_RECORD = 'wn'

# Event names
SPINDLE = 'spindle'
KCOMPLEX = 'kcomplex'

# Metric keys
AF1 = 'af1'
F1_SCORE = 'f1_score'
PRECISION = 'precision'
RECALL = 'recall'
TP = 'tp'
FP = 'fp'
FN = 'fn'
MEAN_ALL_IOU = 'mean_all_iou'
MEAN_NONZERO_IOU = 'mean_nonzero_iou'

# Baselines data keys
F1_VS_IOU = 'f1_vs_iou'
RECALL_VS_IOU = 'recall_vs_iou'
PRECISION_VS_IOU = 'precision_vs_iou'
IOU_HIST_BINS = 'iou_hist_bins'
IOU_CURVE_AXIS = 'iou_curve_axis'
IOU_HIST_VALUES = 'iou_hist_values'
MEAN_IOU = 'mean_iou'
MEAN_AF1 = 'mean_af1'
IQR_LOW_IOU = 'iqr_low_iou'
IQR_HIGH_IOU = 'iqr_high_iou'

# MODEL OUTPUT KEYS
LOGITS = 'logits'
PROBABILITIES = 'probabilities'
SCALOGRAM = 'scalogram'
ATTENTION_WEIGHTS = 'attention_weights'

# Type of padding
PAD_SAME = 'same'
PAD_VALID = 'valid'

# Type of batch normalization
BN = 'bn'
BN_RENORM = 'bn_renorm'

# Type of pooling
MAXPOOL = 'maxpool'
AVGPOOL = 'avgpool'

# Alternative to pooling after convolution
STRIDEDCONV = 'stridedconv'

# Type of dropout
REGULAR_DROP = 'regular_dropout'
SEQUENCE_DROP = 'sequence_dropout'

# Number of directions for recurrent layers
UNIDIRECTIONAL = 'unidirectional'
BIDIRECTIONAL = 'bidirectional'

# Type of class weights
BALANCED = 'balanced'
BALANCED_DROP = 'balanced_drop'
BALANCED_DROP_V2 = 'balanced_drop_v2'

# Types of losses
CROSS_ENTROPY_LOSS = 'cross_entropy_loss'
DICE_LOSS = 'dice_loss'
FOCAL_LOSS = 'focal_loss'
WORST_MINING_LOSS = 'worst_mining_loss'
WORST_MINING_V2_LOSS = 'worst_mining_v2_loss'
CROSS_ENTROPY_NEG_ENTROPY_LOSS = 'cross_entropy_neg_entropy_loss'
CROSS_ENTROPY_SMOOTHING_LOSS = 'cross_entropy_smoothing_loss'
CROSS_ENTROPY_HARD_CLIP_LOSS = 'cross_entropy_hard_clip_loss'
CROSS_ENTROPY_SMOOTHING_CLIP_LOSS = 'cross_entropy_smoothing_clip_loss'
MOD_FOCAL_LOSS = 'mod_focal_loss'
CROSS_ENTROPY_BORDERS_LOSS = 'cross_entropy_borders_loss'
CROSS_ENTROPY_BORDERS_IND_LOSS = 'cross_entropy_borders_ind_loss'
WEIGHTED_CROSS_ENTROPY_LOSS = 'weighted_cross_entropy_loss'
WEIGHTED_CROSS_ENTROPY_LOSS_HARD = 'weighted_cross_entropy_loss_hard'
WEIGHTED_CROSS_ENTROPY_LOSS_SOFT = 'weighted_cross_entropy_loss_soft'
WEIGHTED_CROSS_ENTROPY_LOSS_V2 = 'weighted_cross_entropy_loss_v2'
WEIGHTED_CROSS_ENTROPY_LOSS_V3 = 'weighted_cross_entropy_loss_v3'
WEIGHTED_CROSS_ENTROPY_LOSS_V4 = 'weighted_cross_entropy_loss_v4'
HINGE_LOSS = 'hinge_loss'
WEIGHTED_CROSS_ENTROPY_LOSS_V5 = 'weighted_cross_entropy_loss_v5'  # with anti-borders

# Mix weight strategies
MIX_WEIGHTS_SUM = 'mix_weights_sum'
MIX_WEIGHTS_PRODUCT = 'mix_weights_product'
MIX_WEIGHTS_MAX = 'mix_weights_max'

# Types of optimizer
ADAM_OPTIMIZER = 'adam_optimizer'
ADAM_W_OPTIMIZER = 'adam_w_optimizer'
SGD_OPTIMIZER = 'sgd_optimizer'
RMSPROP_OPTIMIZER = 'rmsprop_optimizer'

# Training params
LOSS_CRITERION = 'loss_criterion'
METRIC_CRITERION = 'metric_criterion'

# Normalization computation mode
NORM_IQR = 'norm_iqr'
NORM_STD = 'norm_std'
NORM_GLOBAL = 'norm_global'

# Colors
RED = 'red'
BLUE = 'blue'
GREEN = 'green'
GREY = 'grey'
DARK = 'dark'
CYAN = 'cyan'
PURPLE = 'purple'

# Model versions
DUMMY = 'dummy'
DEBUG = 'debug'
V1 = 'v1'
V4 = 'v4'
V5 = 'v5'
V6 = 'v6'
V7 = 'v7'
V8 = 'v8'
V9 = 'v9'
V7lite = 'v7lite'
V7litebig = 'v7litebig'
V10 = 'v10'
V11 = 'v11'  # Time-domain
V12 = 'v12'
V13 = 'v13'
V14 = 'v14'
V15 = 'v15'  # Mixed
V16 = 'v16'  # Mixed
V17 = 'v17'
V18 = 'v18'  # Mixed
V19 = 'v19'
V20_INDEP = 'v20_indep'  # time
V20_CONCAT = 'v20_concat'  # time
V21 = 'v21'  # Mixed
V22 = 'v22'  # CWT indep
V23 = 'v23'  # Time-domain with LSTM instead of FC
V24 = 'v24'  # Time-domain with feed-forward UpConv output
V25 = 'v25'  # Time-domain unet
V11_SKIP = 'v11_skip'
V19_SKIP = 'v19_skip'
V19_SKIP2 = 'v19_skip2'
V19_SKIP3 = 'v19_skip3'
V26 = 'v26'  # Experimental skip (based on v19)
V27 = 'v27'  # Experimental skip (based on v11)
V28 = 'v28'  # Experimental skip (based on v11)
V29 = 'v29'  # Experimental skip (based on v11)
V30 = 'v30'  # Experimental skip (based on v11)
V115 = 'v115'  # v11 with kernel 5
V195 = 'v195'  # v19 with kernel 5
V11G = 'v11g'  # v11 with GRU instead of LSTM
V19G = 'v19g'  # V19 with GRU instead of LSTM
V31 = 'v31'  # v19 with independent branches for each band and 2 convs
V32 = 'v32'  # v19 with independent branches for each band and 3 convs
V19P = 'v19p'  # v19 with conv1x1 projection before lstm
V33 = 'v33'  # V19 with independent LSTM's in first layer
V34 = 'v34'  # V19 with 1D convolutions (frequencies as channels)
ATT01 = 'att01'  # 1d attention basic
ATT02 = 'att02'  # 1d attention with lstm
ATT03 = 'att03'  # 1d attention with lstm (better)
ATT04 = 'att04'  # 1d attention with 2 lstm
ATT04C = 'att04c'  # 1d attention with 2 lstm and concat PE.
V35 = 'v35'  # RED-Time+CWT (at last FC)
V11_ABLATION = 'v11_ablation'  # BN and Dropout ablation
V11_ABLATION_SCALED = 'v11_ablation_scaled'  # BN and Dropout ablation with scaled input
V11_D6K5 = 'v11_d6k5'
V11_D8K3 = 'v11_d8k3'
V11_D8K5 = 'v11_d8k5'
V11_OUTRES = 'v11_outres'
V11_OUTPLUS = 'v11_outplus'
V11_SHIELD = 'v11_shield'
V11_LITE = 'v11_lite'
V11_NORM = 'v11_norm'
