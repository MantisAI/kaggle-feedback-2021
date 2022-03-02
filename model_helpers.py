from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification


def load_model(model_name, local_model_path, pretrained_model_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenizer.save_pretrained(local_model_path)
    config_model = AutoConfig.from_pretrained(model_name)
    config_model.num_labels = 15
    config_model.save_pretrained(local_model_path)
    load_model_path = model_name
    if pretrained_model_path:
        load_model_path = pretrained_model_path
    model = AutoModelForTokenClassification.from_pretrained(
        load_model_path, config=config_model
    )

    return model, tokenizer
