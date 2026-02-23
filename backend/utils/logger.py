"""
System event logger - logs to file and database
"""
import logging
import os
from datetime import datetime

LOG_FILE = os.getenv("LOG_FILE", "system_events.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("emotional_support_platform")

def log_event(event_type: str, details: str = ""):
    logger.info(f"[{event_type}] {details}")
