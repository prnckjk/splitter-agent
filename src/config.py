import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")  # Optional
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default to GPT-3.5 Turbo


# File paths
INPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "data", "input")
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "data", "output")

# Ensure output directory exists
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)