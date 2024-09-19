import time
import random
import openai


#Retry with exponential backoff to overcome rate limiting of requests
def call_with_retry(model, prompt, texts, max_retries=5, batch_size=5) -> str:
    """Batch based api call to LLM.
    This is to overcome rate limiting imposed by the model APIs.
    Using an exponential backoff for retries

    Args:
        model (_type_): Model for API call(Ex: OpenAI models)
        prompt (_type_): prompt template
        texts (_type_): Inputs
        max_retries (int, optional): # of retries. Defaults to 5.
        batch_size (int, optional): batch size for API call. Defaults to 5.

    Raises:
        Exception: if max_retries reached 

    Returns:
        str: model response text
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        retries = 0
        while retries < max_retries:
            try:
                # Call the model
                response = model.batch(texts,{"max_concurrency": 1})
                results.extend(response)
                break
            except (openai.RateLimitError,) as e:
                retries += 1
                wait_time = (2 ** retries) + random.uniform(0, 1)  # Exponential backoff
                print(f"Rate limit reached. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        else:
            raise Exception("Maximum retries reached. Please try again later.")
    return results
