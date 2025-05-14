"""
Stores Chain-of-Thought (CoT) related prompt templates for essay (constructed response) questions.
"""

COHERENT_CFA_COT = """
You are a Chartered Financial Analyst (CFA) charterholder. Your task is to answer an essay-style constructed response question from the CFA curriculum. Follow these steps:

1. Restate the question in your own words to clarify the key task or concept being tested.
2. Identify the relevant concepts, formulas, or frameworks from the CFA curriculum that apply to the question.
3. Work through the solution step by step, providing a detailed explanation and calculations where appropriate. Use bullet points or numbered steps to structure your reasoning if helpful.
4. Be clear, concise, and accurate in your explanations, ensuring your response reflects the depth and professionalism expected of a CFA charterholder.
5. Finish with a summary statement that directly answers the question posed.

Vignette:
{vignette}

Question:
{question_stem}

Answer:
"""


SELF_CONSISTENCY_INSTRUCTIONS = """
Repeat the above chain-of-thought essay response N times (with temperature > 0) to get multiple independently reasoned drafts.  
Then compare them to extract the most accurate, complete, and CFA-aligned response—favoring consistency with curriculum frameworks and clarity in financial reasoning.
"""