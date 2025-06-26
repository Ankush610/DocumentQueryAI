from pydantic import validate_call
import logging
import os 

@validate_call
def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with specific name
    """
    logs_dir = './logs'
    os.makedirs(logs_dir,exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel("DEBUG")

    console_handler = logging.StreamHandler()
    console_handler.setLevel("DEBUG")

    file_handler = logging.FileHandler(os.path.join(logs_dir, f"{name}.log"))
    file_handler.setLevel("DEBUG")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
