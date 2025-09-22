# LM Studio client wrapper
# pip install openai tenacity

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

DEFAULT_BASE_URL = "http://localhost:1234/v1"

class ModelNotLoadedError(Exception):
    pass

def create_client(base_url: str = DEFAULT_BASE_URL, api_key: str = "not-needed"):
    return OpenAI(base_url=base_url, api_key=api_key)

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
)
def gen_abstract(client, model: str, prompt: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write formal academic abstracts."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # LM Studio returns 400 if model not loaded or context too small
        raise e