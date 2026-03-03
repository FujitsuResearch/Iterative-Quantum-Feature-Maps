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