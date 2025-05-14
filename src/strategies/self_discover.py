"""
Strategy implementation for the Self-Discover prompting technique for essay generation.

This strategy guides the LLM to first generate a reasoning plan before writing the essay.
"""
import logging
import time
import sys
import threading
from typing import List, Dict, Any
from ..llm_clients import get_llm_response
from ..prompts import self_discover as self_discover_prompt
from ..utils import ui_utils 
from ..utils.prompt_utils import parse_question_data 

logger = logging.getLogger(__name__)

def check_for_refusal(text: str) -> tuple[bool, str | None]:
    """Checks if the response contains any refusal phrases."""
    refusal_phrases = [
        "cannot answer",
        "no options",
        "incomplete question",
        "not provided",
        "unable to determine",
        "cannot provide a response",
        "question is unclear",
        "lack sufficient information",
        "insufficient information",
        "cannot write an essay",
        "unable to write",
        "cannot complete this essay",
        "not enough information to write",
        "cannot generate an essay",
        "unable to generate a response",
        "cannot provide an essay",
        "unable to provide an essay"
    ]
    
    for phrase in refusal_phrases:
        if phrase.lower() in text.lower():
            return True, phrase
    return False, None

def run_self_discover_strategy(essay_data: List[Dict[str, Any]], model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Runs the Self-Discover prompting strategy on a list of essay prompts.

    Args:
        essay_data: A list of dictionaries, each representing an essay prompt.
        model_config: Configuration dictionary for the LLM API call.

    Returns:
        A list of dictionaries, each containing the original essay data
        plus the LLM's response, essay text, refusal status, timing, etc.
    """
    results = []
    total_essays = len(essay_data)
    config_id = model_config.get("config_id", model_config.get("model_id"))
    
    loading_animation = None
    for thread in threading.enumerate():
        if hasattr(thread, '_target') and thread._target and 'LoadingAnimation' in str(thread._target):
            frame = sys._current_frames().get(thread.ident)
            if frame:
                loading_animation = next((lv for lv in frame.f_locals.values() if isinstance(lv, ui_utils.LoadingAnimation)), None)
            if loading_animation: break
    
    if loading_animation: loading_animation.start()
    for i, essay_item in enumerate(essay_data):
        if loading_animation:
            loading_animation.update_progress(i + 1, total_essays)
            loading_animation.message = f"Processing Essay {i+1}/{total_essays} with {config_id} (Self-Discover)"

        try:
            start_time = time.time() 
            prompt = self_discover_prompt.format_self_discover_prompt(essay_item)
            model_type = model_config.get('type')            
            response_data = get_llm_response(prompt, model_config, is_json_response_expected=False) 
            
            if not response_data:
                 logger.error(f"API call failed for essay {i+1}. response_data is None.")
                 
                 end_time = time.time()
                 results.append({
                     **essay_item,
                     'essay_text': "",
                     'is_refusal': True,
                     'refusal_reason': "API call failed",
                     'response_time': end_time - start_time, 
                     'error': "API call failed"
                 })
                 continue  
            response_time = response_data.get('response_time')
            if response_time is None: 
                end_time = time.time()
                response_time = end_time - start_time
            llm_response_text = response_data.get('raw_response_text', 'ERROR: No raw response text found') 
            input_tokens = response_data.get('input_tokens')
            output_tokens = response_data.get('output_tokens')
            
            is_refusal, refusal_reason = check_for_refusal(llm_response_text)
            essay_text = "" if is_refusal else llm_response_text.strip()
            
            results.append({
                **essay_item,
                'essay_text': essay_text,
                'is_refusal': is_refusal,
                'refusal_reason': refusal_reason,
                'response_time': response_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'essay_length': len(essay_text)
            })
        except Exception as e:            
            end_time = time.time() 
            elapsed_time = end_time - start_time if 'start_time' in locals() else 0
            logger.error(f"Error processing essay {i+1} with Self-Discover strategy: {e}", exc_info=True)
            results.append({
                **essay_item,
                'essay_text': "",
                'is_refusal': True,
                'refusal_reason': f"Error: {str(e)}",
                'response_time': elapsed_time, 
                'error': str(e)
            })

    if loading_animation: loading_animation.stop()
    logger.info(f"Self-Discover strategy completed for {len(results)} items using {config_id}.")
    return results 