GEMINI_PROMPT_TEMPLATE = """
I need you to read a vignette and an essay question from the CFA exam and produce a structured essay response.

Instructions:
1. Read the vignette and essay question very carefully.
2. Organize your answer into exactly three clearly labeled sections:
   - Introduction
   - Analysis
   - Conclusion
3. Use concise, professional language appropriate for a CFA essay exam.
4. Address every part of the question directly.
5. Output MUST be a single JSON object with a key "essay" whose value is the full essay text—and NOTHING ELSE. Any deviation will cause a parsing failure.

Vignette:
{vignette}

Essay Question:
{question_stem}

Answer (output JSON):
"""


DEFAULT_PROMPT_TEMPLATE = """
Your task is to answer the following CFA essay question.
**Your response MUST be divided into three headings—Introduction, Analysis, and Conclusion—and include no other sections or commentary.**

Vignette:
{vignette}

Essay Question:
{question_stem}

Answer:
"""

