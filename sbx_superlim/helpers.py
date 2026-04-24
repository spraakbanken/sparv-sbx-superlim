from sparv import api

logger = api.get_logger(__name__)


def get_label_mapper(ds_config, model_config):
    logger.debug("ds_config=%s", ds_config)
    logger.debug("model_config=%s", model_config)
    logger.debug("ds_config.features=%s", ds_config.features)
    ds_labels = ds_config.features["label"].names
    return {model_config.id2label[n]: ds_labels[n] for n in range(len(ds_labels))}
