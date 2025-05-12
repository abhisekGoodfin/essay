"""Metrics calculation functions for evaluating LLM responses."""

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any, Optional

def compute_cosine_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two text strings using TF-IDF.
    Handles empty inputs gracefully.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Cosine similarity score (0.0 to 1.0), or 0.0 if inputs are invalid.
    """
    if not text1 or not text2 or not isinstance(text1, str) or not isinstance(text2, str):
         logging.warning("Invalid input for cosine similarity (empty or non-string). Returning 0.")
         return 0.0

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        texts = [text1, text2]
        tfidf_matrix = vectorizer.fit_transform(texts)

        if tfidf_matrix.shape[1] == 0:
            logging.warning("Vocabulary empty after TF-IDF vectorization. Texts might contain only stop words. Returning 0 similarity.")
            return 0.0

        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0])
    except ValueError as ve:
        logging.warning(f"ValueError during cosine similarity calculation: {ve}. Returning 0.")
        return 0.0
    except Exception as e:
        logging.error(f"Error computing cosine similarity: {e}. Returning 0.")
        return 0.0

def calculate_cost(usage: Dict[str, int], model_type: str, model_id: str, api_costs: Dict[str, Dict[str, Dict[str, float]]]) -> float:
    """
    Estimate the cost of an API call based on token usage and predefined costs.

    Args:
        usage: Dictionary with 'prompt_tokens' and 'completion_tokens'. Can be None or empty.
        model_type: The type of model provider (e.g., 'openai', 'bedrock', 'groq', 'writer', 'xai').
        model_id: The specific model ID.
        api_costs: Dictionary containing API costs per million tokens.

    Returns:
        Estimated cost in USD, or 0.0 if cost info is missing or usage is invalid.
    """
    if not usage or not isinstance(usage, dict):
        return 0.0

    input_tokens = usage.get('prompt_tokens', 0) or 0
    output_tokens = usage.get('completion_tokens', 0) or 0

    provider_type = model_type.lower()
    costs = api_costs.get(provider_type, {}).get(model_id)

    if costs and isinstance(costs, dict):
        input_cost_per_mil = costs.get("input", 0)
        output_cost_per_mil = costs.get("output", 0)

        if input_cost_per_mil is None or output_cost_per_mil is None:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * input_cost_per_mil
        output_cost = (output_tokens / 1_000_000) * output_cost_per_mil

        return input_cost + output_cost
    else:
        cost_warn_key = f"{provider_type}_{model_id}"
        if cost_warn_key not in calculate_cost.warned_models:
             calculate_cost.warned_models.add(cost_warn_key)
             logging.warning(f"Cost information missing or invalid in api_costs for {provider_type}/{model_id}. Returning 0 cost for this model.")
        return 0.0

# Initialize the set of warned models
calculate_cost.warned_models = set()

def evaluate_similarity(results_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate similarity between student answers and model explanations.
    Calculates individual and average similarity scores.

    Args:
        results_data: List of detailed results dictionaries from `process_questions_with_model`.

    Returns:
        Dictionary containing list of similarities, average, and std dev similarity.
    """
    similarities = []
    valid_comparisons = 0

    logging.info("Evaluating similarity between student answers and model explanations...")
    for i, result in enumerate(results_data):
        student_answer = result.get("student_answer", "")
        model_explanation = result.get("model_explanation", "")

        if isinstance(student_answer, str) and isinstance(model_explanation, str) and student_answer and model_explanation and "ERROR:" not in student_answer:
            similarity = compute_cosine_similarity(student_answer, model_explanation)
            similarities.append(similarity)
            result["similarity_score"] = similarity
            valid_comparisons += 1
        else:
            similarities.append(None)
            result["similarity_score"] = None
            logging.debug(f"  Skipping similarity for item {i+1}: Invalid/missing text.")

    valid_similarities = [s for s in similarities if s is not None]
    avg_similarity = float(np.mean(valid_similarities)) if valid_similarities else 0.0
    std_dev_similarity = float(np.std(valid_similarities)) if valid_similarities else 0.0

    logging.info(f"Calculated similarity for {valid_comparisons}/{len(results_data)} items. Average: {avg_similarity:.4f}, StdDev: {std_dev_similarity:.4f}")

    return {
        "average_similarity": avg_similarity,
        "std_dev_similarity": std_dev_similarity,
        "all_similarities": similarities
    } 