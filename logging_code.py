import logging
import sys

def setup_logging(script_name):
    try:
        logger = logging.getLogger(script_name)
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            # Create a file handler for the script
            handler = logging.FileHandler(f'C:\\Users\\golla\\Downloads\\churn\\logs\\{script_name}.log', mode='w')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

        return logger
    except Exception as e:
        err_type, err_msg, err_line = sys.exc_info()
        logger.info(f'Error in lineno {err_line.tb_lineno} and error is due to {err_type} and the error message is {err_msg}')
