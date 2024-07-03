from datasets import get_dataset_config_info

def get_label_mapper(task : str, model_config):
    ds_config = get_dataset_config_info('sbx/superlim-2', task)
    print(ds_config)
    ds_labels = ds_config.features['label'].names
    return {model_config.id2label[n]: ds_labels[n] for n in range(len(ds_labels))}