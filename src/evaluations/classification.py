import logging
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import json

logger = logging.getLogger(__name__)

def get_embeddings(text: str, client: OpenAI) -> List[float]:
    """Get embeddings for a text using text-embedding-3-large model."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        return []

def get_gpt4_judge_score(model_answer: str, llm_answer: str, client: OpenAI) -> Dict[str, Any]:
    """Get GPT-4's evaluation of the answer quality."""
    prompt = f"""You are an expert evaluator. Please evaluate the quality of the LLM's answer compared to the model answer.
    Rate the answer on a scale of 0-10 and provide a brief explanation.

    Model Answer: {model_answer}
    LLM Answer: {llm_answer}

    Please provide your evaluation in JSON format with the following structure:
    {{
        "score": <score between 0-10>,
        "explanation": "<brief explanation of the score>"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error getting GPT-4 evaluation: {e}")
        return {"score": 0, "explanation": f"Error in evaluation: {str(e)}"}

def evaluate_classification(results_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates essay responses using GPT-4 as a judge and text-embedding-3-large for similarity.

    Args:
        results_data: A list of dictionaries, where each dictionary contains
                      at least 'correctAnswer' and 'LLM_answer'.

    Returns:
        A dictionary containing evaluation metrics:
        'average_similarity', 'std_dev_similarity', 'average_judge_score',
        'judge_scores', 'similarity_scores', 'evaluation_details'
    """
    if not results_data:
        logger.warning("No results data provided for evaluation.")
        return {
            "average_similarity": 0.0,
            "std_dev_similarity": 0.0,
            "average_judge_score": 0.0,
            "judge_scores": [],
            "similarity_scores": [],
            "evaluation_details": []
        }

    client = OpenAI()
    similarities = []
    judge_scores = []
    evaluation_details = []

    for item in results_data:
        model_answer = item.get('correctAnswer', '')
        llm_answer = item.get('LLM_answer', '')

        if not isinstance(model_answer, str) or not isinstance(llm_answer, str):
            logger.warning(f"Skipping item due to invalid answer types: {item}")
            continue

        if not model_answer or not llm_answer:
            logger.warning(f"Skipping item due to empty answers: {item}")
            continue

        # Get embeddings and calculate similarity
        model_embedding = get_embeddings(model_answer, client)
        llm_embedding = get_embeddings(llm_answer, client)

        if model_embedding and llm_embedding:
            similarity = cosine_similarity([model_embedding], [llm_embedding])[0][0]
            similarities.append(similarity)
        else:
            similarities.append(0.0)
            logger.warning("Could not calculate similarity due to embedding errors")

        # Get GPT-4 evaluation
        judge_result = get_gpt4_judge_score(model_answer, llm_answer, client)
        judge_scores.append(judge_result['score'])
        
        evaluation_details.append({
            'model_answer': model_answer,
            'llm_answer': llm_answer,
            'similarity_score': similarities[-1],
            'judge_score': judge_result['score'],
            'judge_explanation': judge_result['explanation']
        })

    # Calculate statistics
    valid_similarities = [s for s in similarities if s is not None]
    valid_judge_scores = [s for s in judge_scores if s is not None]

    avg_similarity = float(np.mean(valid_similarities)) if valid_similarities else 0.0
    std_dev_similarity = float(np.std(valid_similarities)) if valid_similarities else 0.0
    avg_judge_score = float(np.mean(valid_judge_scores)) if valid_judge_scores else 0.0

    metrics = {
        "average_similarity": avg_similarity,
        "std_dev_similarity": std_dev_similarity,
        "average_judge_score": avg_judge_score,
        "judge_scores": judge_scores,
        "similarity_scores": similarities,
        "evaluation_details": evaluation_details
    }

    logger.info(f"Evaluation completed. Average similarity: {avg_similarity:.4f}, Average judge score: {avg_judge_score:.2f}")
    return metrics
