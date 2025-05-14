"""
Self-consistency prompting strategy for essay generation.
"""
import logging
import random
import sys
import threading
from collections import Counter

from .. import llm_clients 
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

def generate_prompt_for_essay_strategy(entry: dict, essay_template: str) -> str:
    """Generates an essay prompt for a given entry using the standardized parser."""
    parsed_data = parse_question_data(entry)
    
    return essay_template.format(
        vignette=parsed_data['vignette'],
        question_stem=parsed_data['question_stem']
    )

def run_self_consistency_strategy(data: list[dict], model_config_item: dict, essay_template: str, n_samples: int = 3) -> list[dict]:
    """
    Processes essay prompts using multiple samples and combines them.
    """
    results_for_all_essays = []
    total_essays = len(data)
    config_id = model_config_item.get("config_id", model_config_item.get("model_id"))
    model_type = model_config_item.get("type") 
    sampling_params = model_config_item.get("parameters", {}).copy()
    sampling_params['temperature'] = sampling_params.get('temperature_sampling', 0.7) 
    
    if 'max_tokens_essay' in sampling_params:
        sampling_params['max_tokens'] = sampling_params['max_tokens_essay']
    elif model_config_item.get("type") != "bedrock" or "meta" not in model_config_item.get("model_id", ""):
        sampling_params['max_tokens'] = 4000
    
    sampling_params.pop('response_format', None)

    loading_animation = None
    for thread in threading.enumerate():
        if hasattr(thread, '_target') and thread._target and 'LoadingAnimation' in str(thread._target):
            frame = sys._current_frames().get(thread.ident)
            if frame:
                loading_animation = next((lv for lv in frame.f_locals.values() if isinstance(lv, ui_utils.LoadingAnimation)), None)
            if loading_animation: break

    for i, entry in enumerate(data):
        if loading_animation:
            loading_animation.update_progress(i + 1, total_essays)
            loading_animation.message = f"Processing Essay {i+1}/{total_essays} with {config_id} (Samples)"

        logger.info(f"Processing Essay {i + 1}/{total_essays} with {config_id} (Self-Consistency, {n_samples} samples)...")
        prompt_for_entry = generate_prompt_for_essay_strategy(entry, essay_template)

        sample_responses = [] 
        all_essays_this_prompt = []
        all_refusals = []

        for s_idx in range(n_samples):
            if loading_animation:
                loading_animation.message = f"Processing Essay {i+1}/{total_essays} with {config_id} (Sample {s_idx+1}/{n_samples})"
            
            logger.info(f"  Sample {s_idx + 1}/{n_samples} for Essay {i+1}...")
            
            current_call_model_config = model_config_item.copy()
            current_call_model_config["parameters"] = sampling_params
            
            llm_call_data = llm_clients.get_llm_response(prompt_for_entry, current_call_model_config, is_json_response_expected=False)

            if llm_call_data and not llm_call_data.get("error_message"):
                raw_text = llm_call_data.get('raw_response_text', '')
                is_refusal, refusal_reason = check_for_refusal(raw_text)
                
                if not is_refusal:
                    all_essays_this_prompt.append(raw_text.strip())
                
                sample_responses.append({
                    "essay_text": raw_text.strip(),
                    "is_refusal": is_refusal,
                    "refusal_reason": refusal_reason,
                    "raw_text": raw_text,
                    "response_time": llm_call_data.get('response_time'),
                    "input_tokens": llm_call_data.get('input_tokens'),
                    "output_tokens": llm_call_data.get('output_tokens'),
                })
                
                if is_refusal:
                    all_refusals.append(refusal_reason)
            else:
                logger.error(f"    Failed to get valid response for sample {s_idx + 1} of Essay {i+1}: {llm_call_data.get('error_message', 'Unknown error') if llm_call_data else 'No response'}")
                sample_responses.append({
                    "essay_text": "",
                    "is_refusal": True,
                    "refusal_reason": "API Error",
                    "raw_text": llm_call_data.get('raw_response_text', 'ERROR_NO_RAW_TEXT') if llm_call_data else "ERROR_NO_RESPONSE",
                    "response_time": llm_call_data.get('response_time'),
                    "input_tokens": llm_call_data.get('input_tokens'),
                    "output_tokens": llm_call_data.get('output_tokens'),
                    "error": llm_call_data.get('error_message', 'Unknown error') if llm_call_data else 'No response'
                })
                all_refusals.append("API Error")

        # Determine if we have a valid essay or if all samples were refusals
        final_essay = ""
        is_refusal = False
        refusal_reason = None
        
        if all_essays_this_prompt:
            # If we have valid essays, use the longest one as it's likely the most complete
            final_essay = max(all_essays_this_prompt, key=len)
            logger.info(f"  Essay {i+1} ({config_id}) - Selected longest essay of length {len(final_essay)}")
        else:
            # If all samples were refusals, use the most common refusal reason
            if all_refusals:
                refusal_counter = Counter(all_refusals)
                refusal_reason = refusal_counter.most_common(1)[0][0]
                is_refusal = True
                logger.warning(f"  Essay {i+1} ({config_id}) - All samples were refusals. Most common reason: {refusal_reason}")
            else:
                logger.error(f"  Essay {i+1} ({config_id}) - All {n_samples} samples failed to produce a response.")
                is_refusal = True
                refusal_reason = "All samples failed"

        avg_response_time = sum(s.get('response_time', 0) for s in sample_responses if s.get('response_time')) / len(sample_responses) if sample_responses else 0
        total_input_tokens = sum(s.get('input_tokens', 0) for s in sample_responses if s.get('input_tokens') is not None)
        total_output_tokens = sum(s.get('output_tokens', 0) for s in sample_responses if s.get('output_tokens') is not None)

        logger.info(f"Essay {i+1} ({config_id}) Self-Consistency: Final Essay Length: {len(final_essay)}, Is Refusal: {is_refusal}, Avg Time/Sample: {avg_response_time:.2f}s")

        updated_entry = entry.copy()
        updated_entry.update({
            'essay_text': final_essay,
            'is_refusal': is_refusal,
            'refusal_reason': refusal_reason,
            'response_time': avg_response_time * n_samples, 
            'avg_response_time_per_sample': avg_response_time,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'essay_length': len(final_essay),
            'prompt_strategy': f'self_consistency_n{n_samples}',
            'all_samples_details': sample_responses 
        })
        results_for_all_essays.append(updated_entry)

    if loading_animation: 
        loading_animation.message = f"Processing with {config_id}"

    return results_for_all_essays 