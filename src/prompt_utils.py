"""
Utility functions for processing and formatting prompt data.
"""
import re
from typing import Dict, Any

def parse_question_data(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses the raw question data into standardized components for essay prompt formatting.

    Args:
        question_data: The dictionary for a single question item
                       (expected keys: 'vignette', 'question', 'correctAnswer').

    Returns:
        A dictionary containing:
        'vignette': The vignette text (or default).
        'full_question': The original full question text (for reference).
        'question_stem': The main question text.
        'correct_answer': The correct answer text.
    """
    vignette = question_data.get('vignette', 'No vignette provided.')
    full_question = question_data.get('question', 'No question text provided.')
    correct_answer = question_data.get('correctAnswer', 'No correct answer provided.')

    # For essay questions, the question stem is the full question
    question_stem = full_question.strip()

    return {
        "vignette": vignette,
        "full_question": full_question,
        "question_stem": question_stem,
        "correct_answer": correct_answer
    } 