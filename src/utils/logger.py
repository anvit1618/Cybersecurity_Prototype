import logging
from utils.config import load_config

cfg = load_config()["logging"]

logging.basicConfig(
    filename=cfg["log_file"],
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=getattr(logging, cfg["log_level"])
)

def get_logger(name):
    return logging.getLogger(name)
