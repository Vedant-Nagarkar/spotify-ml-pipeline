import logging
import os
from datetime import datetime

# ── Config ────────────────────────────────────
LOG_DIR  = "logs"
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

os.makedirs(LOG_DIR, exist_ok=True)

# ── Create Logger ─────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "[ %(asctime)s ] %(levelname)s %(name)s - %(message)s",
    handlers= [
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SpotifyML")

# ── Test ──────────────────────────────────────
if __name__ == "__main__":
    logger.info("Logger initialized successfully!")
    logger.info(f"Log file: {LOG_PATH}")
    logger.warning("This is a warning!")
    logger.error("This is an error!")