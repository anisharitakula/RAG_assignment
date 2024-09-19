from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

#prompt template for summarizing
template_summarize = """You are an assistant tasked with summarizing tables and text. \ 
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt_summarize = ChatPromptTemplate.from_template(template_summarize)




# Prompt template for answering questions
template_response = """Answer the question based only on the following context, which can include text and tables:
If you cannot find the answer, please be open to say "I couldn't find the desired information"
{context}
Question: {query}
"""
prompt_response = ChatPromptTemplate.from_template(template_response)




# Prompt template for synthetic question answer generation
template_synthetic_qna = """
You are a teacher who has to conduct a quiz. Your task is to generate **diverse** question, answer, and {doc_id} triplets.

For a given {doc_id}, create **3 unique** question and answer pairs. The instructions are below and must be followed EXACTLY:

1) It is critical that these question-answer pairs are **only** related to the content in the context provided by the USER.
2) The questions should be domain-specific and should not be answerable without the specific context provided by the USER.
3) Ensure **diversity** in the questions. The questions **must not be identical or ask the same thing repeatedly**. They should cover different aspects of the provided content or rephrase the content in different ways.
4) You **MUST ALWAYS RETURN** a triplet of (question, answer, doc_id) as the output. Do NOT add any reasoning or explanations.
5) If enough information is not available to create a question, return ("NONE", "NONE", doc_id).
6) Do not use the exact words from the context in the question. Use semantically equivalent phrases to avoid repetition.
7) If the input is too short or unrelated to create a question and answer, you **MUST** return ("NONE", "NONE", doc_id).
8) You **MUST ONLY** return the exact tuple output. **Do not include reasoning, explanations, or any additional text.**
9) Ensure that **questions are diverse** in both phrasing and focus for the same context.

You will be provided with a context {context} and a doc_id {doc_id}.

You are expected to return multiple unique triplets as a list in the following format:

(
    "What is the operating pressure of TK-3413?",
    "The operating pressure is 1.5 bar.",
    doc_id
)

If no question can be formulated, return:

[(
    "NONE", 
    "NONE",
    doc_id
)]

Example:

USER: 
"TK-3413 is a pressure vessel used to store water. It is used in the production of the Ford F-150. The operating pressure is 1.5 bar. It is made of stainless steel."
AI:
[(
    "What is the operating pressure of TK-3413?",
    "The operating pressure is 1.5 bar.",
    doc_id
)
(
    "What material is TK-3413 made of?",
    "TK-3413 is made of stainless steel.",
    doc_id
)
(
    "Where is TK-3413 used?",
    "It is used in the production of the Ford F-150.",
    doc_id
)]
"""

prompt_qna=ChatPromptTemplate.from_template(template_synthetic_qna)
