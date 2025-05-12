"""Prompt templates for different LLM interactions."""

ORIGINAL_ANSWER_PROMPT_TEMPLATE = """
SYSTEM: You are tasked with answering an essay question from the CFA Level 3 examination.
Read the following vignette and question carefully.
Provide a detailed answer based *only* on the information presented in the vignette and the question.

VIGNETTE:
{case}

QUESTION:
{question}

Your output MUST be a JSON object containing only the key "answer" with your detailed response as the value.
Do not include any text before or after the JSON object. For example:
{{
    "answer": "<Your detailed, comprehensive answer based on the vignette and question>"
}}
"""

ORIGINAL_GRADING_PROMPT_TEMPLATE = """
SYSTEM: You are an expert grader for the CFA Level 3 examination.
Your task is to evaluate a student's essay answer based *strictly* on the provided model answer/explanation (grading details).
Assign a numerical score (e.g., out of 10, assuming 10 is the maximum possible score based on the question's implicit value, but use your judgment if not specified) and provide a concise justification for the score.
The justification should highlight how the student's answer aligns with or deviates from the key points in the model answer.

MODEL ANSWER / GRADING DETAILS:
{answer_grading_details}

STUDENT'S ANSWER:
{answer}

Your output MUST be a JSON object containing the keys "marks" (as a string representing the numerical score) and "explanation" (your justification).
Do not include any text before or after the JSON object. For example:
{{
    "marks": "7",
    "explanation": "The student correctly identified the primary factors but missed the nuance regarding X, as detailed in point 3 of the model answer."
}}
"""

JPM_ANSWER_PROMPT_TEMPLATE = """
SYSTEM : You are taking a test
for the Chartered Financial
Analyst ( CFA ) program
designed to evaluate your
knowledge of different topics
in finance .
You will be given an open ended
essay question . Think step by
step and respond with your
thinking and answer the
question .

Output ONLY the final answer text, without any introductory phrases or preamble.

USER : Case :
{case}
Question :
{question}
Answer :
"""

UNIVERSAL_GRADING_PROMPT_TEMPLATE = """
SYSTEM : You are tasked with
grading essay answers from
the CFA Level 3 examination .
You will be supplied with an
explanation of the correct
answer , the grading details
( where to assign marks ) and
the student 's answer .
Return a numeric value
indicating the number of
marks the student should
receive and the explanation
as to why the student did / did
not receive the marks outline
in the grading detail .
USER : Here are the answer
grading details :
{answer_grading_details}
USER : Here is the student 's
answer :
{answer}
""" 