# Chunking + map-reduce summarization
# pip install tiktoken
import math

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text or ""))
except Exception:
    _enc = None
    def count_tokens(text: str) -> int:
        # fallback estimate
        return max(1, len((text or "").split()) // 0.75)

GEN_PROMPT = (
    "Write a formal academic abstract (~200 words) as one paragraph. "
    "Cover motivation, method, and key results clearly.\n\n{}"
)

def chunk_text(text: str, max_input_tokens: int = 2500, overlap_tokens: int = 200):
    # leave room for prompt and output in a 4k context model
    if not text:
        return []
    words = text.split()
    # rough words per token estimate if no tokenizer
    if _enc is None:
        approx_ratio = 0.75
        step = int(max_input_tokens / approx_ratio)
        ov = int(overlap_tokens / approx_ratio)
        i = 0
        while i < len(words):
            j = min(len(words), i + step)
            yield " ".join(words[i:j])
            i = j - ov if j - ov > i else j
        return
    # token-accurate splitting
    ids = _enc.encode(text)
    i = 0
    while i < len(ids):
        j = min(i + max_input_tokens, len(ids))
        chunk = _enc.decode(ids[i:j])
        yield chunk
        i = j - overlap_tokens if j - overlap_tokens > i else j

def map_reduce(client, model: str, body: str, gen_fn, max_input_tokens=2500):
    # first pass
    chunks = list(chunk_text(body, max_input_tokens=max_input_tokens, overlap_tokens=200))
    if not chunks:
        return ""
    first_pass = []
    for ch in chunks:
        prompt = GEN_PROMPT.format(ch)
        out = gen_fn(client, model, prompt)
        first_pass.append(out)

    # reduce in batches to avoid overflow
    reduced = first_pass
    while len(reduced) > 1:
        next_level = []
        for i in range(0, len(reduced), 6):
            batch = "\n\n".join(reduced[i:i+6])
            prompt = (
                "Combine the following partial abstracts into one coherent paragraph (~200 words). "
                "Focus on motivation, methods, and main results. Avoid bullets.\n\n" + batch
            )
            out = gen_fn(client, model, prompt)
            next_level.append(out)
        reduced = next_level
    return reduced[0]
