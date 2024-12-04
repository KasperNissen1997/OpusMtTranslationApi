"""Handler for the opus-mt models."""

import logging
import json

from transformers import MarianTokenizer, MarianMTModel, pipeline
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class OpusMTHandler(BaseHandler):
    """Handler class for opus-mt models. (MariantMT models, trained on the Opus dataset)"""

    def __init__(self):
        """Initialize class."""
        super(OpusMTHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """Load the hugging face pipeline."""
        model_dir = ctx.system_properties.get("model_dir")

        self.hf_pipeline = pipeline(
            "translation",
            model=MarianMTModel.from_pretrained(model_dir),
            tokenizer=MarianTokenizer.from_pretrained(model_dir, truncation=True, padding=False),
            truncation=True,
        )

        self.initialized = True

    def preprocess(self, data):
        # Log incoming data
        logging.info(f"Incoming data: {data}")

        # Extract the string and pass it to the model
        input_text = data[0]['body']

        logging.info(f"Preprocessed data: {input_text}")

        return input_text

    def inference(self, inputs):
        return self.hf_pipeline(inputs)

    def postprocess(self, data):
        return data