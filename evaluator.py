"""Main evaluation logic for processing questions and grading answers."""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from llm_client import LLMClient
from metrics import compute_cosine_similarity, calculate_cost
from prompts import (
    ORIGINAL_ANSWER_PROMPT_TEMPLATE,
    ORIGINAL_GRADING_PROMPT_TEMPLATE,
    JPM_ANSWER_PROMPT_TEMPLATE,
    UNIVERSAL_GRADING_PROMPT_TEMPLATE
)

class Evaluator:
    """Main evaluator class for processing questions and grading answers."""

    def __init__(self, llm_client: LLMClient, config: Dict[str, Any]):
        """Initialize the evaluator with LLM client and configuration."""
        self.llm_client = llm_client
        self.config = config
        self.results_dir = config.get("RESULTS_DIR", "./results")
        self.charts_dir = config.get("CHARTS_DIR", "./results/charts")
        self.save_interval = config.get("SAVE_INTERVAL", 10)
        self.request_explanations = config.get("REQUEST_EXPLANATIONS", True)

    def process_questions_with_model(self, questions: List[Dict[str, Any]], model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a list of questions with a specific model.
        
        Args:
            questions: List of question dictionaries
            model_config: Model configuration dictionary
            
        Returns:
            List of results dictionaries
        """
        results = []
        total_questions = len(questions)
        
        logging.info(f"\nProcessing {total_questions} questions with model {model_config['config_id']}")
        
        for i, question in enumerate(questions, 1):
            try:
                result = self._process_single_question(question, model_config)
                results.append(result)
                
                # Save intermediate results
                if i % self.save_interval == 0:
                    self._save_intermediate_results(results, model_config['config_id'])
                    
            except Exception as e:
                logging.error(f"Error processing question {i}: {e}")
                results.append({
                    "question_id": question.get("id", f"unknown_{i}"),
                    "error": str(e)
                })
                
        return results

    def _process_single_question(self, question: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single question with the given model configuration."""
        start_time = time.time()
        
        try:
            # Generate answer
            answer_prompt = self._format_answer_prompt(question)
            answer_response, answer_usage = self.llm_client.call_model(
                model_config["type"],
                model_config["model_id"],
                answer_prompt,
                model_config["parameters"]
            )
            
            # Parse answer
            try:
                answer_data = json.loads(answer_response)
                student_answer = answer_data.get("answer", "")
            except json.JSONDecodeError:
                student_answer = answer_response
                
            # Grade answer if grading details are available
            grade_result = None
            if "answer_grading_details" in question:
                grade_prompt = self._format_grading_prompt(question, student_answer)
                grade_response, grade_usage = self.llm_client.call_model(
                    model_config["type"],
                    model_config["model_id"],
                    grade_prompt,
                    model_config["parameters"]
                )
                
                try:
                    grade_data = json.loads(grade_response)
                    grade_result = {
                        "marks": grade_data.get("marks", "0"),
                        "explanation": grade_data.get("explanation", "")
                    }
                except json.JSONDecodeError:
                    grade_result = {
                        "marks": "0",
                        "explanation": "Error parsing grading response"
                    }
                    
                # Calculate similarity if explanations are requested
                similarity_score = None
                if self.request_explanations and grade_result:
                    similarity_score = compute_cosine_similarity(
                        student_answer,
                        grade_result["explanation"]
                    )
                    
            # Calculate costs
            answer_cost = calculate_cost(
                answer_usage,
                model_config["type"],
                model_config["model_id"],
                self.config.get("API_COSTS", {})
            )
            
            grade_cost = 0.0
            if grade_result:
                grade_cost = calculate_cost(
                    grade_usage,
                    model_config["type"],
                    model_config["model_id"],
                    self.config.get("API_COSTS", {})
                )
                
            elapsed_time = time.time() - start_time
            
            return {
                "question_id": question.get("id", "unknown"),
                "student_answer": student_answer,
                "grade_result": grade_result,
                "similarity_score": similarity_score,
                "answer_usage": answer_usage,
                "grade_usage": grade_usage if grade_result else None,
                "answer_cost": answer_cost,
                "grade_cost": grade_cost,
                "total_cost": answer_cost + grade_cost,
                "elapsed_time": elapsed_time
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logging.error(f"Error processing question after {elapsed_time:.2f}s: {e}")
            return {
                "question_id": question.get("id", "unknown"),
                "error": str(e),
                "elapsed_time": elapsed_time
            }

    def _format_answer_prompt(self, question: Dict[str, Any]) -> str:
        """Format the answer prompt based on the question type."""
        if question.get("prompt_type") == "jpm":
            return JPM_ANSWER_PROMPT_TEMPLATE.format(
                case=question.get("case", ""),
                question=question.get("question", "")
            )
        else:
            return ORIGINAL_ANSWER_PROMPT_TEMPLATE.format(
                case=question.get("case", ""),
                question=question.get("question", "")
            )

    def _format_grading_prompt(self, question: Dict[str, Any], student_answer: str) -> str:
        """Format the grading prompt based on the question type."""
        if question.get("prompt_type") == "jpm":
            return UNIVERSAL_GRADING_PROMPT_TEMPLATE.format(
                answer_grading_details=question.get("answer_grading_details", ""),
                answer=student_answer
            )
        else:
            return ORIGINAL_GRADING_PROMPT_TEMPLATE.format(
                answer_grading_details=question.get("answer_grading_details", ""),
                answer=student_answer
            )

    def _save_intermediate_results(self, results: List[Dict[str, Any]], model_id: str) -> None:
        """Save intermediate results to a JSON file."""
        if not self.config.get("SAVE_INTERMEDIATE_RESULTS", True):
            return
            
        try:
            output_file = Path(self.results_dir) / f"intermediate_results_{model_id}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Saved intermediate results to {output_file}")
        except Exception as e:
            logging.error(f"Error saving intermediate results: {e}")

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results across all questions for a model.
        
        Args:
            results: List of results dictionaries from process_questions_with_model
            
        Returns:
            Dictionary containing aggregated statistics
        """
        if not results:
            return {}
            
        total_questions = len(results)
        successful_answers = sum(1 for r in results if "error" not in r)
        successful_grades = sum(1 for r in results if r.get("grade_result") is not None)
        
        total_time = sum(r.get("elapsed_time", 0) for r in results)
        total_cost = sum(r.get("total_cost", 0) for r in results)
        
        total_input_tokens = sum(
            r.get("answer_usage", {}).get("prompt_tokens", 0) +
            r.get("grade_usage", {}).get("prompt_tokens", 0)
            for r in results if "error" not in r
        )
        
        total_completion_tokens = sum(
            r.get("answer_usage", {}).get("completion_tokens", 0) +
            r.get("grade_usage", {}).get("completion_tokens", 0)
            for r in results if "error" not in r
        )
        
        similarity_scores = [r.get("similarity_score") for r in results if r.get("similarity_score") is not None]
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        
        marks = [float(r.get("grade_result", {}).get("marks", 0)) for r in results if r.get("grade_result")]
        avg_marks = sum(marks) / len(marks) if marks else 0.0
        
        return {
            "total_questions": total_questions,
            "successful_answers": successful_answers,
            "successful_grades": successful_grades,
            "answer_success_rate": successful_answers / total_questions if total_questions > 0 else 0.0,
            "grade_success_rate": successful_grades / total_questions if total_questions > 0 else 0.0,
            "average_similarity": avg_similarity,
            "average_marks": avg_marks,
            "total_time_seconds": total_time,
            "average_time_per_question": total_time / total_questions if total_questions > 0 else 0.0,
            "estimated_total_cost": total_cost,
            "estimated_cost_per_question": total_cost / total_questions if total_questions > 0 else 0.0,
            "total_input_tokens": total_input_tokens,
            "total_completion_tokens": total_completion_tokens,
            "average_tokens_per_question": (total_input_tokens + total_completion_tokens) / total_questions if total_questions > 0 else 0.0
        } 