import logging
import os
from logging.handlers import RotatingFileHandler

class AppLogger:
    def __init__(self, name=__name__, log_file='app.log', overwrite=True):
        """Initialize the logger
        
        Args:
            name (str): Name of the logger (usually __name__)
            log_file (str): Path to the log file
            overwrite (bool): If True, overwrites existing log file (default: True)
        """
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers to prevent duplicate logs
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Set file handler mode
        file_mode = 'w' if overwrite else 'a'
        
        # Create file handler (with overwrite capability)
        file_handler = RotatingFileHandler(
            log_file,
            mode=file_mode,          # 'w' for overwrite, 'a' for append
            maxBytes=1024*1024,       # 1MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return self.logger