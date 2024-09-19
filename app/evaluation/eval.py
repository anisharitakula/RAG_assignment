from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def retriever_eval(retrieved_doc_ids: list , answer_doc_id: str) -> int:
    """Generic retriever evaluation metrics. Calculating Hit rate below
    Hit rate defined as % of queries where the relevant document is retrieved in the top-k results

    Args:
        retrieved_doc_ids (list): List of retrieved doc_ids
        answer_doc_id (str): The relevant doc_id

    Returns:
        int: if relevant doc_id is present
    """

    if answer_doc_id in retrieved_doc_ids:
        return 1
    return 0

def generator_eval(generated_response: str , answer: str) -> float:
    """Generic generator evaluation metrics. Using cosine similarity below to 
    calculate similarity between generated response and expected response

    Args:
        generated_response (str): 
        answer (str): 

    Returns:
        float: Cosine similarity between generated and expoected response 
    """

    # Load a pre-trained Sentence Transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Encode sentences to get their embeddings
    embeddings = model.encode([generated_response, answer])

    # Compute cosine similarity
    cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return cosine_sim