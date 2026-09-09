import torch
import tiktoken

from llms.configs import get_config
from llms.models import GPTModel
from llms.gpt_download import download_and_load_gpt2

import llms.utils as utils




class LLMOchestrator():

    def __init__(self, config_name):
        self.model_config = get_config(config_name)
        self.model = GPTModel(self.model_config)        
        self.device = utils.get_device()
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def load_weight(self):
        settings, params = download_and_load_gpt2(
            self.model_config["model_size"],
            "./gpt2")
                
        utils.load_weights_into_gpt(self.model, params)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, 
                                              map_location=self.device, 
                                              weights_only=True))

    def create_model(self, model_path, config=None):
        pass

    def get_config(self, model_name):
        pass

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def generate_text(self, start_context, max_new_tokens):
        token_ids = utils.generate_text_simple(
            model=self.model,
            max_new_tokens=max_new_tokens,
            idx=utils.text_to_token_ids(start_context, self.tokenizer),
            context_size=self.model_config["context_length"]
        )

        return utils.token_ids_to_text(token_ids, self.tokenizer)
    



    
