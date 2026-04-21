def _config_candidates(config):
    yield config

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        yield text_config


def get_hidden_size(config):
    for candidate in _config_candidates(config):
        for attr in ("n_embd", "hidden_size", "d_model"):
            value = getattr(candidate, attr, None)
            if value is not None:
                return value
    raise NotImplementedError(
        f"Unable to determine hidden size from config type {type(config).__name__}"
    )


def get_max_position_embeddings(config):
    for candidate in _config_candidates(config):
        for attr in (
            "n_positions",
            "max_sequence_length",
            "max_position_embeddings",
            "seq_length",
        ):
            value = getattr(candidate, attr, None)
            if value is not None:
                return value
    raise NotImplementedError(
        f"Unable to determine max position embeddings from config type {type(config).__name__}"
    )
