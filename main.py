import logging
import os
import platform
import sys

from mrunner.helpers.client_helper import get_configuration

from src.data import build_dataset
from src.model import build_model
from src.trainer import train

# StreamHandler flushes per record; this covers third-party prints (HF's metric dicts)
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=f"[%(levelname)s][host:{platform.node()}]"
    f"[local_rank:{os.environ.get('LOCAL_RANK')}] %(message)s",
)
# HF hub downloads log one INFO line per HTTP request via httpx; keep warnings only
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# model_name -> params.model.name
params = get_configuration(nesting_prefixes=("model_", "data_", "train_"))
logger.info(f"params: {params}")

model, tokenizer = build_model(params.model)
dataset = build_dataset(params.data, tokenizer)
train(model, tokenizer, dataset, params.train)
