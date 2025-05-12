"""Main entry point for the CFA essay evaluation system."""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

from config import Config
from llm_client import LLMClient
from evaluator import Evaluator
from utils import setup_environment, setup_directories, get_credential
from visualization import generate_comparison_charts

def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_questions(input_file: str) -> List[Dict[str, Any]]:
    """Load questions from a JSON file."""
    try:
        with open(input_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading questions from {input_file}: {e}")
        raise

def get_api_keys() -> Dict[str, str]:
    """Get API keys from environment variables or user input."""
    api_keys = {}
    
    # List of required API keys
    required_keys = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "WRITER_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY"
    ]
    
    for key in required_keys:
        value = get_credential(key)
        if value:
            api_keys[key] = value
        else:
            logging.warning(f"API key {key} not found in environment variables")
            
    return api_keys

def get_selected_models(config: Config) -> List[Dict[str, Any]]:
    """Get the list of selected models to run."""
    if config.SELECTED_MODELS:
        return [m for m in config.MODEL_CONFIGS if m["config_id"] in config.SELECTED_MODELS]
        
    selected = []
    if config.RUN_ANTHROPIC_MODELS:
        selected.extend([m for m in config.MODEL_CONFIGS if m["type"] == "anthropic"])
    if config.RUN_BEDROCK_MODELS:
        selected.extend([m for m in config.MODEL_CONFIGS if m["type"] == "bedrock"])
    if config.RUN_OPENAI_MODELS:
        selected.extend([m for m in config.MODEL_CONFIGS if m["type"] == "openai"])
    if config.RUN_GEMINI_MODELS:
        selected.extend([m for m in config.MODEL_CONFIGS if m["type"] == "gemini"])
    if config.RUN_GROQ_MODELS:
        selected.extend([m for m in config.MODEL_CONFIGS if m["type"] == "groq"])
    if config.RUN_WRITER_MODELS:
        selected.extend([m for m in config.MODEL_CONFIGS if m["type"] == "writer"])
    if config.RUN_XAI_MODELS:
        selected.extend([m for m in config.MODEL_CONFIGS if m["type"] == "xai"])
        
    return selected

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="CFA Essay Evaluation System")
    parser.add_argument("--input", required=True, help="Input JSON file containing questions")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        # Set up environment
        setup_environment()
        
        # Load configuration
        config = Config()
        
        # Set up directories
        setup_directories(config.RESULTS_DIR, config.CHARTS_DIR)
        
        # Load questions
        questions = load_questions(args.input)
        logging.info(f"Loaded {len(questions)} questions from {args.input}")
        
        # Get API keys
        api_keys = get_api_keys()
        if not api_keys:
            logging.error("No API keys found. Please set the required API keys in environment variables.")
            return
            
        # Initialize LLM client
        llm_client = LLMClient(api_keys)
        
        # Initialize evaluator
        evaluator = Evaluator(llm_client, config.__dict__)
        
        # Get selected models
        selected_models = get_selected_models(config)
        if not selected_models:
            logging.error("No models selected. Please configure model selection in config.py")
            return
            
        logging.info(f"Selected {len(selected_models)} models to run")
        
        # Process questions with each model
        all_results = []
        for model_config in selected_models:
            try:
                results = evaluator.process_questions_with_model(questions, model_config)
                aggregated = evaluator.aggregate_results(results)
                
                # Add model info to aggregated results
                aggregated["Model"] = model_config["config_id"]
                aggregated["PromptSet"] = "Original"  # Default prompt set
                
                all_results.append(aggregated)
                
                # Save detailed results
                output_file = Path(config.RESULTS_DIR) / f"detailed_results_{model_config['config_id']}.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logging.info(f"Saved detailed results to {output_file}")
                
            except Exception as e:
                logging.error(f"Error processing model {model_config['config_id']}: {e}")
                
        # Save comparison summary
        if all_results:
            summary_file = Path(config.RESULTS_DIR) / config.COMPARISON_SUMMARY_FILE
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logging.info(f"Saved comparison summary to {summary_file}")
            
            # Generate comparison charts
            try:
                import pandas as pd
                df = pd.DataFrame(all_results)
                generate_comparison_charts(df, config.CHARTS_DIR)
            except Exception as e:
                logging.error(f"Error generating comparison charts: {e}")
                
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 