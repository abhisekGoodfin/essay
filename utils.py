"""Utility functions for the CFA essay evaluation system."""

import os
import logging
import subprocess
from pathlib import Path
from typing import Optional

def setup_environment() -> None:
    """
    Install required packages using pip. Ensures necessary libraries are available.
    Includes plotting libraries needed for advanced visualization.
    """
    logging.info("Setting up environment...")
    try:
        packages = [
            "python-dotenv", "reportlab",
            "openai", "scikit-learn", "boto3", "groq", "writerai", "anthropic",
            "google-generativeai>=0.4.0",
            "plotly", "pandas", "kaleido"
        ]

        subprocess.check_call(["pip", "install", "--upgrade"] + packages)
        logging.info("Required packages installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing packages: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during environment setup: {e}")

def setup_directories(results_dir: str, charts_dir: str) -> None:
    """
    Create necessary directories for final results/charts.
    Ensures the script doesn't fail due to missing output paths.

    Args:
        results_dir: Directory for storing results
        charts_dir: Directory for storing charts
    """
    logging.info("Setting up directories...")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(charts_dir).mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directories: {results_dir}, {charts_dir}")

def get_credential(key: str, default: str = "") -> str:
    """
    Get a credential from Google Colab userdata or environment variables.
    Handles potential errors during credential access.

    Args:
        key: The key name of the credential
        default: Default value if credential is not found

    Returns:
        The credential value or default
    """
    try:
        try:
            from google.colab import userdata
            IS_COLAB = True
            logging.info("Running in Google Colab environment, using userdata for secrets.")
        except ImportError:
            IS_COLAB = False
            logging.info("Not running in Colab environment, using environment variables for secrets.")

        if IS_COLAB:
            value = userdata.get(key)
            return value if value is not None else default
        else:
            return os.environ.get(key, default)
    except Exception as e:
        logging.warning(f"Error getting credential {key}: {e}. Using default.")
        return default 