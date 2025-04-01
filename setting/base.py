import os
import json
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# LLM settings
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "aya-expanse")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Database settings with fallback to None for empty strings
DATABASE_URI = os.environ.get("DATABASE_URI")
if DATABASE_URI == "":
    DATABASE_URI = None
logger.info(f"Using database URI: {DATABASE_URI}")

# Ensure SESSION_POOL_SIZE is an integer
SESSION_POOL_SIZE = int(os.environ.get("SESSION_POOL_SIZE", 40))


# Model configurations
def parse_model_configs() -> dict:
    """Parse MODEL_CONFIGS from environment variable"""
    config_str = os.environ.get("MODEL_CONFIGS", "{}")
    try:
        return json.loads(config_str)
    except json.JSONDecodeError:
        print(f"Warning: Invalid MODEL_CONFIGS format: {config_str}")
        return {}


MODEL_CONFIGS = parse_model_configs()
