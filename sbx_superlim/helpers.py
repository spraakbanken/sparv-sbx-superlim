def get_label_mapper(ds_config, model_config):
    ds_labels = ds_config.features['label'].names
    return {model_config.id2label[n]: ds_labels[n] for n in range(len(ds_labels))}