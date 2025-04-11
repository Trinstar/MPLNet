from mplnet.lib.boolean_flags import boolean_flags
from mplnet.lib.count_parameters import count_parameters
from mplnet.lib.float_range import float_range
from mplnet.lib.get_clip_hyperparams import get_clip_hyperparams
from mplnet.lib.get_params_group import get_params_group
from mplnet.lib.get_set_random_state import get_random_state, set_random_state, get_set_random_state, random_seed
from mplnet.lib.ood_metrics import get_fpr, get_auroc
from mplnet.lib.json_utils import save_json, load_json
from mplnet.lib.load_checkpoint import load_checkpoint
from mplnet.lib.log_ood_metrics import log_ood_metrics
from mplnet.lib.logger import LOGGER, setup_logger
from mplnet.lib.meters import AverageMeter, DictAverage, ProgressMeter
from mplnet.lib.save_checkpoint import save_checkpoint
from mplnet.lib.track import track


__all__ = [
    "boolean_flags",
    "count_parameters",
    "float_range",
    "get_clip_hyperparams",
    "get_params_group",
    "get_random_state",
    "set_random_state",
    "get_set_random_state",
    "random_seed",
    "get_fpr",
    "get_auroc",
    "save_json",
    "load_json",
    "load_checkpoint",
    "log_ood_metrics",
    "LOGGER",
    "setup_logger",
    "AverageMeter",
    "DictAverage",
    "ProgressMeter",
    "save_checkpoint",
    "track",
]


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
