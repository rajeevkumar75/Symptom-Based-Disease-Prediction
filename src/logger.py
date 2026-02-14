import logging
import os
import sys
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(levelname)s %(name)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("symptom_disease_logger")


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message
        _, _, exc_tb = error_detail.exc_info()
        self.line_no = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return (
            f"Error occurred in file [{self.file_name}] "
            f"at line [{self.line_no}] "
            f"with message [{self.error_message}]"
        )
