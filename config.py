from logging.config import dictConfig


logging_config = {
    "version": 1,

    "disable_existing_loggers": False,

    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(name)s] %(levelname)-4s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    },

    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}

dictConfig(logging_config)


USER_ITEM_INTERACTIONS_DATA_PATH = "data/user-item-interactions"