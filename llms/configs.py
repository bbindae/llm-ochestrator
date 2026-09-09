model_configs = {
    "gpt2-small (124M)": {"model_name": "gpt2-small", "model_size": "124M", "emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"model_name": "gpt2-medium", "model_size": "355M", "emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"model_name": "gpt2-large", "model_size": "774M", "emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"model_name": "gpt2-xl", "model_size": "1558M", "emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}


def get_config(config_name):
    config = BASE_CONFIG    
    if config_name == "gpt2-small":
        config.update(model_configs["gpt2-small (124M)"])
    elif config_name == "gpt2-medium":
        config.update(model_configs["gpt2-medium (355M)"])
    elif config_name == "gpt2-large (774M)":
        config.update(model_configs["gpt2-large (774M)"])
    elif config_name == "gpt2-xl (1558M)":
        config.update(model_configs["gpt2-xl (1558M)"])
    else:
        raise ValueError("Invalid config name")

    return config