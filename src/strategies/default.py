"""
Default processing strategy: process each essay prompt once.
"""
import logging
import sys
import threading
from .. import llm_clients 
from ..utils import ui_utils
from ..utils.prompt_utils import parse_question_data
from ..prompts import default as default_prompts

logger = logging.getLogger(__name__)

def generate_prompt_for_default_strategy(entry: dict, model_type: str | None = None) -> str:
    """Generates the default prompt for the LLM using the standardized parser."""
    parsed_data = parse_question_data(entry)
    
    template = default_prompts.DEFAULT_PROMPT_TEMPLATE
    if model_type == "gemini":
        template = default_prompts.GEMINI_PROMPT_TEMPLATE
        
    return template.format(
        vignette=parsed_data['vignette'],
        question_stem=parsed_data['question_stem']
    )

def run_default_strategy(data: list[dict], model_config_item: dict) -> list[dict]:
    """
    Processes each essay prompt using the specified LLM, expecting a full essay response.
    Uses the standardized prompt generation.
    """
    results = []
    total_questions = len(data)
    config_id = model_config_item.get("config_id", model_config_item.get("model_id"))
    model_type = model_config_item.get("type")
    
    loading_animation = None
    
    for thread in threading.enumerate():
        if hasattr(thread, '_target') and thread._target and 'LoadingAnimation' in str(thread._target):
            frame = sys._current_frames().get(thread.ident)
            if frame:
                loading_animation = next((lv for lv in frame.f_locals.values() if isinstance(lv, ui_utils.LoadingAnimation)), None)
            if loading_animation: 
                break

    for i, entry in enumerate(data):
        if loading_animation:
            loading_animation.update_progress(i + 1, total_questions)
            loading_animation.message = f"Processing Essay {i+1}/{total_questions} with {config_id} (Default Strategy)"

        logger.info(f"Processing essay {i + 1}/{total_questions} with model {config_id} (Default Strategy)...")
        
        prompt = generate_prompt_for_default_strategy(entry, model_type=model_type)
        
        llm_data = llm_clients.get_llm_response(prompt, model_config_item)

        essay_text = ""
        model_answer = ""
        is_refusal = False
        refusal_reason = None
        response_time = None
        current_input_tokens = None
        current_output_tokens = None
        essay_length = 0
        raw_response_text = None
        error_message = None

        if llm_data:
            response_time = llm_data.get('response_time')
            current_input_tokens = llm_data.get('input_tokens')
            current_output_tokens = llm_data.get('output_tokens')
            raw_response_text = llm_data.get('raw_response_text')
            error_message = llm_data.get('error_message')
            
            response_content = llm_data.get('response_content', {})
            if isinstance(response_content, dict):
                essay_text = response_content.get('essay', '')
                model_answer = response_content.get('model_answer', '')
                is_refusal = response_content.get('is_refusal', False)
                refusal_reason = response_content.get('refusal_reason')
                error_message = response_content.get('error')
            
            essay_length = len(essay_text)

        updated_entry = entry.copy()
        updated_entry.update({
            'essay_text': essay_text,
            'model_answer': model_answer,
            'is_refusal': is_refusal,
            'refusal_reason': refusal_reason,
            'response_time': response_time,
            'input_tokens': current_input_tokens,
            'output_tokens': current_output_tokens,
            'essay_length': essay_length,
            'raw_response_text': raw_response_text,
            'error': error_message,
            'prompt_strategy': 'default'
        })
        results.append(updated_entry)

    if loading_animation: 
        loading_animation.stop()
        
    return results 