import inspect
import torch
import torch_npu
from pathlib import Path
from typing import Type, List
from transformers import AutoModel, AutoModelForSequenceClassification
from opentelemetry import trace

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Prediction, TokenEmbedding

tracer = trace.get_tracer(__name__)


class RerankModel(Model):
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(dtype).to(device).eval()

        self.has_position_ids = (
            inspect.signature(model.forward).parameters.get("position_ids", None)
            is not None
        )
        self.has_token_type_ids = (
            inspect.signature(model.forward).parameters.get("token_type_ids", None)
            is not None
        )

        super(RerankModel, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        print("rerank model is not support embed")
    
    @tracer.start_as_current_span("embed_all")
    def embed_all(self, batch: PaddedBatch) -> List[TokenEmbedding]:
        print("rerank model is not support embed_all")

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Prediction]:
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        if self.has_token_type_ids:
            kwargs["token_type_ids"] = batch.token_type_ids
        if self.has_position_ids:
            kwargs["position_ids"] = batch.position_ids

        scores = self.model(**kwargs).logits.cpu().tolist()

        return [
            Prediction(
                values=scores[i]
            )
            for i in range(len(batch))
        ]
