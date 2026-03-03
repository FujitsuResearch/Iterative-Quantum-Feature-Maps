import logging

def get_module_logger(modname, filename, level='info'):
    logger = logging.getLogger(modname)
    if level == 'debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    filehandler = logging.FileHandler(filename, 'a+')
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    return logger

import os
def get_logger_parent_folder(logger):
    """
    Retrieve the parent folder of the log file used by a logger.
    
    Args:
        logger (logging.Logger): The logger instance.
    
    Returns:
        str: The parent folder of the log file, or None if no FileHandler is found.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):  # Check if the handler is a FileHandler
            log_file_path = handler.baseFilename       # Full path to the log file
            return os.path.dirname(log_file_path), os.path.basename(handler.baseFilename)      # Parent folder of the log file
    return None  # No FileHandler found