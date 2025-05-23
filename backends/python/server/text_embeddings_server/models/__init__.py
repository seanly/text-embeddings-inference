import os
import torch
import torch_npu
from loguru import logger
from pathlib import Path
from typing import Optional
from transformers import AutoConfig
from transformers.models.bert import BertConfig

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.masked_model import MaskedLanguageModel
from text_embeddings_server.models.default_model import DefaultModel
from text_embeddings_server.models.rerank_model import RerankModel

from modeling_bert_adapter import enable_bert_speed
from modeling_roberta_adapter import enable_roberta_speed
from modeling_xlm_roberta_adapter import enable_xlm_roberta_speed

__all__ = ["Model"]

# Disable gradients
torch.set_grad_enabled(False)

FLASH_ATTENTION = True
try:
    from text_embeddings_server.models.flash_bert import FlashBert
except ImportError as e:
    logger.warning(f"Could not import Flash Attention enabled models: {e}")
    FLASH_ATTENTION = False

if FLASH_ATTENTION:
    __all__.append(FlashBert)


def get_model(model_path: Path, dtype: Optional[str], pool: str):
    if dtype == "float32":
        dtype = torch.float32
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype {dtype}")

    enable_boost = os.getenv("ENABLE_BOOST", "False")
    if enable_boost not in("True", "False"):
        raise ValueError("env ENABLE_BOOST value must be True or False")
    
    if enable_boost == "True":
        dtype == torch.float16
            
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.npu.is_available():
        device = torch.device("npu")
        torch.npu.set_compile_mode(jit_compile=False)
        option = {"NPU_FUZZY_COMPILE_BLACKLIST": "ReduceProd"}
        torch.npu.set_option(option)
        deviceIdx = os.environ.get('TEI_NPU_DEVICE')
        if deviceIdx != None and deviceIdx.isdigit() and int(deviceIdx) >= 0 and int(deviceIdx) <= 7:
            torch.npu.set_device(torch.device(f"npu:{deviceIdx}"))
    else:
        if dtype != torch.float32:
            raise ValueError("CPU device only supports float32 dtype")
        device = torch.device("cpu")

    config = AutoConfig.from_pretrained(model_path)

    if config.architectures[0].endswith("Classification"):
        return RerankModel(model_path, device, dtype)
    else:
        if (
            config.model_type == "bert"
            and device.type == "cuda"
            and config.position_embedding_type == "absolute"
            and dtype in [torch.float16, torch.bfloat16]
            and FLASH_ATTENTION
        ):
            return FlashBert(model_path, device, dtype)
        elif config.architectures[0].endswith("ForMaskedLM"):
            return MaskedLanguageModel(model_path, device, dtype, pool)
        else:
            return DefaultModel(model_path, device, dtype, pool)
        

    raise NotImplementedError
