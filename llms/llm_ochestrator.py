import torch
import tiktoken

from llms.configs import get_config
from llms.models import GPTModel,GPTClassificationModel
from llms.gpt_download import download_and_load_gpt2

import llms.data_loaders as data_loaders
import llms.utils as utils
import llms.utils_classifier as utils_classifier



class LLMOchestrator():

    def __init__(self, config_name, model_type):
        self.model_config = get_config(config_name)
        self.device = utils.get_device()
        self.model_type = model_type
        self.model = self.__get_model()              
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.optimizer = self.__get_optimizer()

    def __get_model(self):
        if self.model_type == "basic":            
            return GPTModel(self.model_config).to(self.device)
        elif self.model_type == "classification":            
            return GPTClassificationModel(cfg=self.model_config).to(self.device)
        else:
            raise ValueError("invalid model type")

    def __get_optimizer(self):
        if self.model_type == "basic":
            return torch.optim.AdamW(self.model.parameters(), lr=0.0004, weight_decay=0.1)
        elif self.model_type == "classification":
            return torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.1)
       
    def load_weight(self):
        settings, params = download_and_load_gpt2(
            self.model_config["model_size"],
            "./gpt2")
                
        utils.load_weights_into_gpt(self.model, params)

    def save_model(self, model_path):
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "optimizer_class": self.optimizer.__class__.__name__
                },
                model_path)   


    def load_model(self, model_path):
        checkpoint = torch.load(model_path,
                                weights_only=False,
                                map_location=self.device)

        
        self.model.load_state_dict(checkpoint["model_state_dict"])
    
        opt_state = checkpoint["optimizer_state_dict"]
        opt_class_name = checkpoint.get("optimizer_class", "AdamW")
        opt_class = getattr(torch.optim, opt_class_name)

        param_group = opt_state["param_groups"][0]
        saved_lr = param_group.get("lr", 0.0004)
        saved_weight_decay = param_group.get("weight_decay", 0.1)

        self.optimizer = opt_class(self.model.parameters(), 
                                   lr=saved_lr,
                                   weight_decay=saved_weight_decay)  

        # TODO: write code when "otimizer_clas is not available"              
        

    def create_model(self, model_path, config=None):
        pass

    def get_config(self, model_name):
        pass

    def save_model(self, model_path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_class": self.optimizer.__class__.__name__
            },
            model_path)

    def generate_text(self, start_context, max_new_tokens, temperature=0.0, top_k=None, eos_id=None):
        #when eos_id encountered, stop
        token_ids = utils.generate_text(
            model=self.model.to(self.device),
            max_new_tokens=max_new_tokens,
            idx=utils.text_to_token_ids(start_context, self.tokenizer).to(self.device),
            context_size=self.model_config["context_length"],
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id
        )

        return utils.token_ids_to_text(token_ids, self.tokenizer)

    def create_plantext_dataloader(self, file_path, train_ratio=0.7, validation_ratio=0.2, batch_size=4, shuffle=True, drop_last=True, num_workers=0):
                
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

        train_end = int(len(text_data) * train_ratio)
        train_data = text_data[:train_end]
        train_loader = data_loaders.create_plantext_dataloader(train_data, self.tokenizer,
                                                               batch_size=batch_size, max_length=self.model_config["context_length"],
                                                               stride=self.model_config["context_length"], shuffle=shuffle, drop_last=False,
                                                               num_workers=num_workers)

        validation_end = train_end + int(len(text_data) * validation_ratio)
        validation_data = text_data[train_end:validation_end]
        validation_loader = data_loaders.create_plantext_dataloader(validation_data, self.tokenizer,
                                                                    batch_size=batch_size, max_length=self.model_config["context_length"],
                                                                    stride=self.model_config["context_length"], shuffle=False, drop_last=False,
                                                                    num_workers=num_workers)      

        test_loader = None

        if train_ratio + validation_ratio < 1:
            test_data = text_data[validation_end:]
            test_loader = data_loaders.create_plantext_dataloader(test_data, self.tokenizer,
                                                                    batch_size=batch_size, max_length=self.model_config["context_length"],
                                                                    stride=self.model_config["context_length"], shuffle=False, drop_last=drop_last,
                                                                    num_workers=num_workers)
               

        return train_loader, validation_loader, test_loader

    def train_model_by_plantext(self, num_epochs=10, eval_freq=5, eval_iter=5,start_context="Hello World", 
                                train_loader=None, validation_loader=None, batch_size = None, file_path=None,
                                warmup_steps_ratio=0.2, initial_lr=3e-05, min_lr=1e-6):

        if train_loader == None or validation_loader == None:
            if batch_size == None:
                raise ValueError("batch_size should be specified")
            train_loader, validation_loader, test_loader = self.create_plantext_dataloader(file_path=file_path, train_ratio=0.7, validation_ratio=0.2,
                                            batch_size=batch_size)

        train_losses, val_losses, tokens_seen, track_lrs = utils.train_model(self.model, train_loader=train_loader, val_loader=validation_loader,
                                 optimizer=self.optimizer, device=self.device, num_epochs=num_epochs,
                                 eval_freq=eval_freq, eval_iter=eval_iter, start_context=start_context,
                                 tokenizer=self.tokenizer, warmup_steps_ratio=warmup_steps_ratio,
                                 initial_lr=initial_lr, min_lr=min_lr)

        self.num_epochs = num_epochs
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.tokens_seen = tokens_seen

        return train_losses, val_losses, tokens_seen, track_lrs

    def finetune_model_by_plantext(self):
        pass

    def plot_losses(self, num_epochs=None, tokens_seen=None, train_losses=None, val_losses=None):

        if num_epochs==None or tokens_seen==None or train_losses==None or val_losses==None:
            num_epochs = self.num_epochs
            tokens_seen = self.tokens_seen
            train_losses = self.train_losses
            val_losses = self.val_losses

        epochs_tensor = torch.linspace(1, num_epochs, len(train_losses))
        utils.plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        
    def create_spamdata_dataloader(self, train_file_path, val_file_path, test_file_path, batch_size=8, shuffle=True, drop_last=True, num_workers=0):
        
        train_loader = data_loaders.create_spamdata_dataloader(
        file_path=train_file_path,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        tokenizer=self.tokenizer
        )

        val_loader = data_loaders.create_spamdata_dataloader(
        file_path=val_file_path,
        batch_size=batch_size,
        num_workers=num_workers,        
        drop_last=False,
        tokenizer=self.tokenizer
        )

        test_loader = data_loaders.create_spamdata_dataloader(
        file_path=test_file_path,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        tokenizer=self.tokenizer
        )

        assert train_loader.dataset.max_length <= self.model_config["context_length"], (
            f"Dataset length {train_loader.dataset.max_length} exceeds model's context "
            f"length {self.model_config['context_length']}. Reinitialize data sets with "
            f"`max_length={self.model_config['context_length']}`"
        )
        return train_loader, val_loader, test_loader

    def train_classifier(self, num_epochs=10, eval_freq=5, eval_iter=10,
                                train_loader=None, validation_loader=None, batch_size = 8, file_path=None,
                                warmup_steps_ratio=0.2,
                                initial_lr=3e-05, min_lr=1e-6):

        if train_loader == None or validation_loader == None:
            if batch_size == None:
                raise ValueError("batch_size should be specified")
            if file_path == None:
                raise ValueError("file_path should be specified")
            train_loader, validation_loader, test_loader = self.create_spamdata_dataloader(
                train_file_path=file_path + "train.csv", 
                val_file_path=file_path+"validation.csv",
                test_file_path=file_path+"test.csv",
                batch_size=batch_size     
            )

        train_losses, val_losses, train_accs, val_accs, examples_seen, track_lrs = utils_classifier.train_classifier(
            model=self.model,
            train_loader=train_loader,
            val_loader=validation_loader,
            optimizer=self.optimizer,
            device=self.device,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
            eval_iter=eval_iter,
            warmup_steps_ratio=warmup_steps_ratio,
            initial_lr=initial_lr,
            min_lr=min_lr            
        )

        self.num_epochs = num_epochs
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.tokens_seen = examples_seen

        return train_losses, val_losses, train_accs, val_accs, examples_seen, track_lrs
        
    def classify_review(self, text, max_length=None, pad_token_id=50256):
        return utils_classifier.classify_review(text=text,
                                model=self.model,
                                tokenizer=self.tokenizer,
                                device=self.device,
                                max_length=max_length,
                                pad_token_id=pad_token_id)

    def plot_values(self, epochs_seen, examples_seen, train_values, val_values, label="loss"):
        utils_classifier.plot_values(epochs_seen, examples_seen,train_values,val_values,label)

    def calc_accuracy_loader(self, data_loader, num_batches=None):
        return utils_classifier.calc_accuracy_loader(data_loader=data_loader, 
                                              model=self.model,
                                              device=self.device,
                                              num_batches=num_batches)

    def calc_loss_loader(self, data_loader, num_batches=None):
        return utils_classifier.calc_loss_loader(data_loader=data_loader,
                                                 model=self.model,
                                                 device=self.device,
                                                 num_batches=num_batches)




    
