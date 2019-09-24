import json
from argparse import ArgumentParser
from typing import List

from util.common import get_logger
from util.directory import fetch_param_path

logger = get_logger(__name__)


EXP_INDEX = 37
TRAINING = False
HIGH_USE_IMAGE = True
USE_HISTORY = True
NOISY_DATA = False
NUM_NOISY_SAMPLES = 3

if TRAINING:
    parser = ArgumentParser()
    parser.add_argument('model_type', type=str)
    parser.add_argument('dataset_version', type=int)
    parser.add_argument('image_type', type=str)
    args = parser.parse_args()

    DATASET_VERSION = args.dataset_version
    model_type = args.model_type
    ABLATION_TYPE = 'none'
    if model_type in ['control', 'stop']:
        USER_NET_TYPE = 'se'
        USER_MODEL_LEVEL = 'low'
        USER_CONTROL = 'true' if model_type == 'control' else 'false'
        USER_ENCODER_TYPE = 'onehot'
    elif model_type.startswith('high'):
        words = model_type.split('-')
        encoder_type = words[1]
        if encoder_type not in ['onehot', 'gru']:
            raise ValueError('encoder type should be specified {}'.format(model_type))
        USER_NET_TYPE = words[2] if len(words) > 2 else 'ga'
        USER_MODEL_LEVEL = 'high'
        USER_CONTROL = 'true'
        USER_ENCODER_TYPE = encoder_type

    # ablation-{single/high}, ablation-nmc-{control,stop,high}
    elif model_type.startswith('ablation'):
        logger.info(model_type)
        words = model_type.split('-')
        ABLATION_TYPE = words[1]
        if not HIGH_USE_IMAGE:
            assert ABLATION_TYPE == 'high'
            USER_NET_TYPE = 'wi'
            USER_MODEL_LEVEL = 'high'
            USER_CONTROL = 'true'
            USER_ENCODER_TYPE = 'gru'
        elif ABLATION_TYPE == 'nmc':
            USER_NET_TYPE = 'nmc'
            assert len(words) == 3
            if words[2] in ['control', 'stop']:
                USER_MODEL_LEVEL = 'low'
                USER_CONTROL = 'true' if words[2] == 'control' else 'false'
                USER_ENCODER_TYPE = 'onehot'
            else:
                USER_MODEL_LEVEL = 'high'
                USER_CONTROL = 'true'
                USER_ENCODER_TYPE = 'gru'
        elif ABLATION_TYPE in ['high', 'single']:
            USER_NET_TYPE = 'ga'
            USER_MODEL_LEVEL = 'high' if ABLATION_TYPE == 'high' else 'low'
            USER_CONTROL = 'true'
            USER_ENCODER_TYPE = 'gru'
        else:
            raise TypeError('invalid ablation type {}'.format(ABLATION_TYPE))
    IMAGE_TYPE = args.image_type
else:
    USER_NET_TYPE = 'se'
    USER_MODEL_LEVEL = 'low'
    USER_CONTROL = 'true'
    USER_ENCODER_TYPE = 'onehot'
    DATASET_VERSION = 37
    IMAGE_TYPE = 'bgr'
    ABLATION_TYPE = 'none'
USER_DECODER_TYPE = 'gru'
USER_MACRO_TYPE = 'large'

# quickly modify frequently tuned parameters
if USER_MACRO_TYPE == 'small':
    MACRO_SIZES = 32, 128
elif USER_MACRO_TYPE == 'large':
    MACRO_SIZES = 64, 256
else:
    raise ValueError('MACRO uses only one of four values: {}'.format(USER_MACRO_TYPE))

SINGLE_LABEL = ''

NET_TYPE = USER_NET_TYPE
MODEL_LEVEL = USER_MODEL_LEVEL
IS_CONTROL = USER_CONTROL == 'true'
ENCODER_TYPE = USER_ENCODER_TYPE
DECODER_TYPE = USER_DECODER_TYPE
# MAX_DATA_LEN = (20 if USER_MODEL_LEVEL == 'low' else 5) if DECODER_TYPE == 'gru' else 1

LOW_LEVEL_SEGMENT = False
SEGMENT_CUT_RATIO = 0.5
INCLUDE_LANE_FOLLOW = False
BALANCE_LABEL = True
LOSS_WEIGHT_TYPE = 'none'
ATTENTION_TYPE = 'luong'

# dimensions of the model
LEN_LOW = 20
LEN_HIGH = 5
LEN_SINGLE = 5
DIM_CONTROL = 2
DIM_STOP = 1
DIM_SUBTASK = 5
if ABLATION_TYPE != 'none':
    if ABLATION_TYPE == 'single':
        ACTION_DIM = DIM_CONTROL
        MAX_DATA_LEN = LEN_SINGLE
        MODEL_IDENTIFIER = 'ablation-single'
    elif ABLATION_TYPE == 'nmc':
        ATTENTION_TYPE = 'none'
        MODEL_IDENTIFIER = 'ablation-nmc'
        if USER_MODEL_LEVEL == 'high':
            MODEL_IDENTIFIER += '-high'
            ACTION_DIM = DIM_SUBTASK
            MAX_DATA_LEN = LEN_HIGH if DECODER_TYPE == 'gru' else 1
        else:
            if IS_CONTROL:
                ACTION_DIM = DIM_CONTROL
                MODEL_IDENTIFIER += '-control'
            else:
                ACTION_DIM = DIM_STOP
                MODEL_IDENTIFIER += '-stop'
            MAX_DATA_LEN = LEN_LOW
    elif ABLATION_TYPE == 'high':
        ACTION_DIM = DIM_SUBTASK
        MAX_DATA_LEN = LEN_HIGH
        MODEL_IDENTIFIER = 'ablation-high'
    else:
        raise TypeError('invalid ablation type {}'.format(ABLATION_TYPE))
    assert DECODER_TYPE == 'gru'
else:
    if MODEL_LEVEL == 'low':
        if IS_CONTROL:
            ACTION_DIM = DIM_CONTROL  # control
            MODEL_IDENTIFIER = 'control'
        else:
            ACTION_DIM = DIM_STOP  # stop
            MODEL_IDENTIFIER = 'stop'
        MAX_DATA_LEN = LEN_LOW if DECODER_TYPE == 'gru' else 1
    else:
        MODEL_IDENTIFIER = 'high'
        ACTION_DIM = DIM_SUBTASK  # high
        MAX_DATA_LEN = LEN_HIGH if DECODER_TYPE == 'gru' else 1

BATCH_SIZE = (5 if MODEL_IDENTIFIER == 'ablation-single' else 64) if DECODER_TYPE == 'gru' else 128
OUT_ENCODER = 32 if ENCODER_TYPE == 'gru' else 16
CONV_SIZE, DECODER_SIZE = MACRO_SIZES
CV_CHANNELS = [len(IMAGE_TYPE), DECODER_SIZE, CONV_SIZE, CONV_SIZE]
IN_DECODER = DECODER_SIZE
OUT_DECODER = DECODER_SIZE

EVAL_KEYWORD = 'left'
EVAL_TIMESTAMP = 1557967689321614  # 1557462636814076
EVAL_ENFORCE_SENTENCE = ''  # 'turn left and then right'


LR_INIT = 3e-4
LR_STEP = 20000
LR_RATE = 1.0 / 2.0

DATASET_DATA_NAMES = ['semantic1']
DATASET_INFO_NAMES = ['semantic1-v{}'.format(DATASET_VERSION)]
DATASET_DATA_PROBS = [1.0]

INJECT_NOISE = False 
USE_MULTI_CAM = True
HAS_CLUTERS = True
USE_SEQUENCE = True
SCHEDULED_SAMPLING = True
SCHEDULED_ITERATION = 200000

TRAIN_DATASET = 'd{}'.format(DATASET_DATA_NAMES[0])
EXP_WORDS = [MODEL_IDENTIFIER, TRAIN_DATASET, IMAGE_TYPE]
if DECODER_TYPE == 'gru':
    EXP_WORDS += ['len{}'.format(MAX_DATA_LEN)]
EXP_WORDS += ['v{}'.format(DATASET_VERSION)]  # temporally used
if not USE_HISTORY:
    EXP_WORDS += ['wh']
if not HIGH_USE_IMAGE:
    EXP_WORDS += ['wi']
if NOISY_DATA:
    EXP_WORDS += ['noisy{}'.format(NUM_NOISY_SAMPLES)]
EXP_NAME = '-'.join(EXP_WORDS)


if TRAINING:
    logger.info('fetched parameter {}'.format(EXP_NAME))
    logger.info('net-type       {}'.format(NET_TYPE))
    logger.info('model-level    {}'.format(MODEL_LEVEL))
    logger.info('max-data-len   {}'.format(MAX_DATA_LEN))
    logger.info('action-dim     {}'.format(ACTION_DIM))
    logger.info('ablation-type  {}'.format(ABLATION_TYPE))
    logger.info('is-control     {}'.format(IS_CONTROL))
    logger.info('attention-type {}'.format(ATTENTION_TYPE))
    logger.info('encoder-type   {}'.format(ENCODER_TYPE))
    logger.info('decoder-type   {}'.format(DECODER_TYPE))
    logger.info('use-image      {}'.format(HIGH_USE_IMAGE))
    logger.info('use-history    {}'.format(USE_HISTORY))
    logger.info('noisy-data     {}'.format(NOISY_DATA))


class ParameterInfo:
    def __init__(self, exp_index: int, exp_name: str):
        self.exp_index = exp_index
        self.exp_name = exp_name


class Parameter:
    def __init__(self):
        self.output_size: int = ACTION_DIM
        self.device_type: str = 'cuda:0'

        # output of word embedding and the input size to encoder (from https://arxiv.org/pdf/1706.07230.pdf)
        self.encoder_embedding_size: int = 50
        self.encoder_bidirectional: bool = False
        self.encoder_layers: int = 1
        self.encoder_cell_type: str = 'gru'
        self.encoder_variable_length: bool = False
        self.encoder_hidden_size: int = OUT_ENCODER
        self.encoder_dropout_probability: float = 0.1
        self.encoder_type = ENCODER_TYPE
        self.encoder_num_words = 100

        self.decoder_bidirectional: bool = False
        self.decoder_layers: int = 1
        self.decoder_cell_type: str = 'gru'
        self.decoder_variable_length: bool = False
        self.decoder_hidden_size: int = OUT_DECODER
        self.decoder_dropout_probability: float = 0.1
        self.decoder_iterative: bool = False
        self.decoder_type = DECODER_TYPE
        self.decoder_rollout = True

        self.cv_visual_dims = CV_CHANNELS  # now it is used in conv layers
        self.out_linear = IN_DECODER  # in_decoder_dim
        self.out_decoder = OUT_DECODER

        self.fc_final_dims = [self.decoder_hidden_size, self.decoder_hidden_size // 4, self.output_size]
        self.onehot_dims = [3, 16, self.encoder_hidden_size]

        self.ablation_type: str = ABLATION_TYPE
        self.update_word_embedding: bool = True
        self.update_visual_embedding: bool = True
        self.model_type: str = NET_TYPE
        self.model_level: str = MODEL_LEVEL
        self.is_control: bool = IS_CONTROL
        self.use_ranging_functions: bool = True
        self.batch_norm_type: str = 'none'
        self.attention_type: str = ATTENTION_TYPE
        self.use_glove_embedding: bool = True
        self.grouped_batch: bool = self.model_type in ['se']
        self.use_history: bool = USE_HISTORY
        self.history_size: int = IN_DECODER // 4
        self.image_type: str = IMAGE_TYPE
        self.high_use_image: bool = HIGH_USE_IMAGE

        self.noisy_data: bool = NOISY_DATA
        self.num_noisy_samples: int = NUM_NOISY_SAMPLES
        self.subtrajectory: bool = False
        self.max_data_length: int = MAX_DATA_LEN
        self.use_low_level_segment: bool = LOW_LEVEL_SEGMENT
        self.high_level_segment_cut_ratio: float = SEGMENT_CUT_RATIO
        self.include_lane_follow: bool = INCLUDE_LANE_FOLLOW
        self.balance_label: bool = BALANCE_LABEL
        self.single_label: str = SINGLE_LABEL
        self.use_multi_cam: bool = USE_MULTI_CAM
        self.has_clusters: bool = HAS_CLUTERS
        self.use_sequence: bool = USE_SEQUENCE
        self.scheduled_sampling: bool = SCHEDULED_SAMPLING
        self.scheduled_iteration: int = SCHEDULED_ITERATION

        # data augmentation (extra cameras)
        self.use_color_jitter: bool = len(IMAGE_TYPE) > 1
        self.color_jitter_range: float = 0.1
        self.color_jitter_probability: float = 0.5

        self.inject_noise: bool = INJECT_NOISE
        self.noise_type: str = 'normal'  # 'linear'
        self.noise_std: float = 0.1
        self.noise_interval: int = 10
        self.noise_duration: int = 2
        self.noise_max_steer: float = 0.1

        self.lr_step: int = LR_STEP
        self.lr_rate: float = LR_RATE
        self.lr_init: float = LR_INIT
        self.loss_type: str = 'l1'
        self.optimizer_type: str = 'adam'
        self.print_every: int = 10
        self.valid_every: int = 1
        self.epoch: int = 0
        self.step: int = 0
        self.max_grad_norm: float = 1.0
        self.num_epoch: int = 200000
        self.loss_weight_type: str = LOSS_WEIGHT_TYPE
        self.loss_weight_min: float = 0.0
        self.loss_weight_max: float = 1.0

        # training data parameter
        self.dataset_data_names: List[str] = DATASET_DATA_NAMES
        self.dataset_info_names: List[str] = DATASET_INFO_NAMES
        self.dataset_data_probs: List[float] = DATASET_DATA_PROBS
        self.random_seed: int = 0
        self.batch_size: int = BATCH_SIZE
        self.num_workers: int = 4
        self.split_train: bool = False
        self.train_ratio: float = 0.9
        # self.num_evaluation_samples: int = 30
        self.shuffle = True
        assert len(self.dataset_data_names) == len(self.dataset_data_probs)

        # experiment
        self.debug: bool = False
        self.exp_name: str = EXP_NAME
        self.exp_index: int = EXP_INDEX
        self.step: int = -1

        # evaluations
        self.eval_keyword: str = EVAL_KEYWORD
        self.eval_timestamp: int = EVAL_TIMESTAMP
        self.eval_data_name: str = ''
        self.eval_info_name: str = ''

    @property
    def path(self):
        return fetch_param_path(self.exp_index, self.exp_name)

    def save(self):
        with open(str(self.path), 'w') as file:
            json.dump(self.__dict__, file, indent=2)

    def load(self):
        with open(str(self.path), 'r') as file:
            loaded_param = json.load(file)
        for key, value in loaded_param.items():
            self.__dict__[key] = value
        logger.info('parameter was loaded from {}'.format(self.path))

    def check(self):
        self.check_batch_norm_type()
        self.check_decoder()

    @property
    def encoder_hidden_layers(self):
        return self.encoder_layers * (2 if self.encoder_bidirectional else 1)

    @property
    def decoder_hidden_layers(self):
        return self.decoder_layers * (2 if self.decoder_bidirectional else 1)

    def check_batch_norm_type(self):
        if self.batch_norm_type == 'sequence':
            if not self.subtrajectory:
                raise TypeError('sequence type of batch norm should use subtrajectory data feed')
        elif self.batch_norm_type not in ['feature', 'none']:
            raise ValueError('invalid batch_norm_type: {}'.format(self.batch_norm_type))

    def check_decoder(self):
        if self.decoder_rollout and self.decoder_type not in ['gru', 'lstm']:
            raise ValueError('decoder rollout requires recurrent decoder')
        if self.attention_type == 'luong' and not self.decoder_rollout:
            raise ValueError('attention luong requires decoder rollout')


def fetch_parameter(exp_index: int, exp_name: str) -> Parameter:
    path = fetch_param_path(exp_index, exp_name)
    if not path.exists():
        raise FileNotFoundError('the path was not found: {}'.format(path))
    param = Parameter()
    param.load()
    param.check()
    return param
