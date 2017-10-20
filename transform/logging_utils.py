'''
Created on Aug 23, 2017

@author: kashefy
'''
import logging

def setup_logging(fpath,
                  name=None, 
                  level=logging.DEBUG):
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(fpath)
    ch = logging.StreamHandler()
    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    for h in [fh, ch]:
        fh.setLevel(level)
        h.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(h)
    return logger
    
def close_logging(logger):
    # remember to close the handlers
    for handler in logger.handlers:
        handler.close()
        logger.removeFilter(handler)
    while logger.handlers:
        logger.handlers.pop()
    