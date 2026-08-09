import logging
import json
from pythonjsonlogger import jsonlogger

def setup_logging(service_name: str):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger = logging.getLogger(service_name)
    return logger
