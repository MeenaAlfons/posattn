def ensure_config(config, defaults):
    """
    Recursively ensures that all keys in defaults are in config.
    Otherwise, it sets the default value.
    """
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            if isinstance(config[key], dict):
                ensure_config(config[key], value)
            else:
                config[key] = value
                print(
                    'WARNING: config[{}] is not a dict, using default value {}'.
                    format(key, value)
                )
        elif config[key] is not None:
            config.update(
                {key: type(value)(config[key])}, allow_val_change=True
            )


def unflatten_dict(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]

        if parts[-1] not in d:
            d[parts[-1]] = value
        else:
            # This will only happen if parts[-1] is a parent to another key.
            # Setting it here will remove the subtree of parts[-1].
            # Therefore, we ignore setting at because this only happens when
            # there is a defuatl value of empty dict for a key and the config
            # has a value to a sub key in that dict.
            pass
    return resultDict
