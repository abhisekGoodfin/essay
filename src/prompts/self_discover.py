"""
Prompt template for the Self-Discover reasoning strategy for essay response questions.

Guides the LLM to first outline a reasoning structure before writing the essay.
Inspired by: https://arxiv.org/abs/2402.14310v1
"""
from ..utils.prompt_utils import parse_question_data

SELF_DISCOVER_PROMPT_TEMPLATE = """\
**Task:** Answer the following essay question by first devising a reasoning structure using the Self-Discover method.

**Context/Vignette:**
{vignette}

**Essay Prompt:**
{question_stem}

**Self-Discover Reasoning Process:**

**1. Select Reasoning Modules:**
   Identify and list the core reasoning modules or thinking strategies needed to address this essay prompt (e.g., Concept Mapping, Argument Structuring, Evidence Integration, Comparative Analysis, Theoretical Application).

**2. Adapt Modules to the Prompt:**
   For each selected module, explain how it applies to this essay and outline the steps you will take within that module.

**3. Implement Reasoning Structure:**
   Execute the plan step-by-step, showing your analytical structure, using bullet points or subheadings to track your thought process.

**4. Essay Response:**
   Based on the structured reasoning above, write a coherent essay with an introduction, developed body paragraphs, and a clear conclusion.

**Begin Reasoning:**

[Your structured reasoning goes here]

**Begin Essay:**

[Your essay response goes here]
"""

def format_self_discover_prompt(question_data: dict) -> str:
    """
    Formats the Self-Discover essay prompt with the specific context and prompt.

    Args:
        question_data: Dictionary containing at least 'vignette' and 'question_stem'.

    Returns:
        Formatted prompt string.
    """
    parsed_data = parse_question_data(question_data)
    return SELF_DISCOVER_PROMPT_TEMPLATE.format(
        vignette=parsed_data['vignette'],
        question_stem=parsed_data['question_stem']
    )

# Optional helper for consistency with MCQ function naming

def generate_prompt_for_self_discover_strategy(entry: dict) -> str:
    """Generates the Self-Discover essay prompt for the LLM."""
    return format_self_discover_prompt(entry)
