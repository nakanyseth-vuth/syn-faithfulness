import os
import logging
from typing import Tuple, List, Dict, Set, Any, Optional
from collections import Counter
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file if present

try:  # vLLM is optional at runtime
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except Exception:  # pragma: no cover - triggered when vllm is not installed/usable
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    _VLLM_AVAILABLE = False

try:  # API fallback via OpenAI-compatible clients is optional
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover - triggered when openai package is absent
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False

try:
    import requests
    _REQUESTS_AVAILABLE = True
except Exception:  # pragma: no cover - triggered when requests is absent
    requests = None  # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False


logger = logging.getLogger(__name__)

_VLLM_CACHE: Dict[str, Any] = {}
_OPENAI_CLIENT: Optional[Any] = None




def chat_vllm(
    model: Optional[Any],
    query: str,
    system_query: str,
    sampling_params: Optional[Any]
) -> str:
    """Generate text using a vLLM backend."""
    if model is None or sampling_params is None:
        raise RuntimeError("vLLM backend is unavailable")

    messages = (
        [{"role": "system", "content": system_query}] if system_query else []
    ) + [{"role": "user", "content": query}]

    try:
        output = model.chat(messages, sampling_params, use_tqdm=False)
    except Exception as exc:  # pragma: no cover - runtime GPU/device failures
        raise RuntimeError(f"vLLM generation failed: {exc}") from exc

    if not output or not getattr(output[0], "outputs", None):
        raise RuntimeError("vLLM returned empty output")

    generated_text = output[0].outputs[0].text
    return generated_text



def load_llm(model_name_or_path: str = "HuggingFaceH4/zephyr-7b-beta") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a Hugging Face causal language model and tokenizer.

    Args:
        model_name_or_path (str, optional): The model name or path from Hugging Face Hub 
            or a local directory. Defaults to "HuggingFaceH4/zephyr-7b-beta".

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: 
            - model: The loaded Hugging Face CausalLM model.
            - tokenizer: The tokenizer corresponding to the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        padding_side="right", 
        use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer


def load_vllm(model_name_or_path: str = "HuggingFaceH4/zephyr-7b-beta") -> Optional[Any]:
    """Load (and cache) a vLLM model for text generation.

    Returns:
        Optional[Any]: A ready-to-use vLLM instance, or ``None`` when vLLM is not
        available or the model fails to load (e.g., due to missing GPU resources).
    """
    if not _VLLM_AVAILABLE or LLM is None:
        logger.debug("vLLM is not available in this environment")
        return None

    if model_name_or_path in _VLLM_CACHE:
        return _VLLM_CACHE[model_name_or_path]

    try:
        model = LLM(model=model_name_or_path, task="generate")
    except Exception as exc:  # pragma: no cover - runtime/device failures
        logger.warning("Failed to load vLLM model '%s': %s", model_name_or_path, exc)
        _VLLM_CACHE[model_name_or_path] = None
        return None

    _VLLM_CACHE[model_name_or_path] = model
    return model


def chat_api(
    query: str,
    system_query: str,
    temperature: float,
    max_tokens: int,
    model_name: Optional[str] = None,
) -> str:
    """Generate text using an OpenAI-compatible API backend."""
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("API key not provided (set OPENAI_API_KEY or LLM_API_KEY)")

    base_url = (
        os.getenv("LLM_API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    target_model = (
        model_name
        or os.getenv("LLM_API_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-3.5-turbo"
    )

    messages = (
        [{"role": "system", "content": system_query}] if system_query else []
    ) + [{"role": "user", "content": query}]

    # Prefer the official OpenAI client when available.
    if _OPENAI_AVAILABLE and OpenAI is not None:
        global _OPENAI_CLIENT
        if _OPENAI_CLIENT is None:
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            _OPENAI_CLIENT = OpenAI(**client_kwargs)

        client = _OPENAI_CLIENT

        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choices = getattr(response, "choices", None)
            if choices:
                return choices[0].message.content or ""
        except AttributeError:
            # The installed SDK does not provide chat.completions (>=1.0 prefers responses)
            pass
        except Exception as exc:
            raise RuntimeError(f"OpenAI chat.completions failed: {exc}") from exc

        try:
            response = client.responses.create(
                model=target_model,
                input=messages,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text
            outputs = getattr(response, "output", None)
            if outputs:
                text_chunks = [getattr(chunk, "text", "") for chunk in outputs]
                combined = "".join(text_chunks).strip()
                if combined:
                    return combined
        except Exception as exc:
            raise RuntimeError(f"OpenAI responses API failed: {exc}") from exc

        raise RuntimeError("OpenAI client returned no content")

    if not _REQUESTS_AVAILABLE or requests is None:
        raise RuntimeError("Neither OpenAI SDK nor requests library is available for API mode")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    endpoint_base = base_url or "https://api.openai.com/v1"
    url = endpoint_base.rstrip("/") + "/chat/completions"

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"HTTP API request failed: {exc}") from exc

    choices = data.get("choices") if isinstance(data, dict) else None
    if choices:
        return choices[0].get("message", {}).get("content", "")

    raise RuntimeError("API response did not contain choices")


def load_redocred() -> Any:
    """
    Load the reDocRED dataset from Hugging Face Datasets.
    Args:
        split (str, optional): The dataset split to load. Defaults to "train".
    Returns:
        Dataset: A Hugging Face Dataset object
    """
    return load_dataset("docred")


def format_dataset_redocred(dataset: Any) -> List[Dict]:
    """
    Format the DocRED dataset into annotated triples.

    Args:
        dataset (Dataset): A Hugging Face Dataset object loaded from DocRED.

    Returns:
        List[Dict]: A list of annotated documents, each containing:
            - "triples" (list[dict]): Extracted triples with head, tail, relation, and evidence.
            - "text" (list[list[str]]): Sentences in the document.
            - "vertexSet" (list[list[dict]]): Original entity mentions and metadata.
            - "doc_idx" (int): Index of the document.
    """
    annotated_triples: List[Dict] = []

    for idx, data in enumerate(dataset):
        triples: List[Dict] = []

        labels = data.get("labels", {})
        vertex_set = data.get("vertexSet", [])

        heads = labels.get("head", [])
        if not heads:  # Skip documents with no labeled relations
            continue

        tails = labels.get("tail", [])
        relations = labels.get("relation_text", [])
        evidences = labels.get("evidence", [])

        for i, (h, t) in enumerate(zip(heads, tails)):
            head_mentions = vertex_set[h] if h < len(vertex_set) else []
            tail_mentions = vertex_set[t] if t < len(vertex_set) else []

            for h_mention in head_mentions:
                for t_mention in tail_mentions:
                    triple = {
                        "head": {
                            "entity": h_mention.get("name", ""),
                            "type": h_mention.get("type", ""),
                        },
                        "tail": {
                            "entity": t_mention.get("name", ""),
                            "type": t_mention.get("type", ""),
                        },
                        "relation": relations[i] if i < len(relations) else "",
                        "evidence": evidences[i] if i < len(evidences) else [],
                    }
                    if triple not in triples:
                        triples.append(triple)

        annotated_triples.append({
            "triples": triples,
            "text": data.get("sents", []),
            "vertexSet": vertex_set,
            "doc_idx": idx,
        })

    return annotated_triples


def format_triples_for_input(triples: List[Dict[str, Dict[str, str]]]) -> str:
    """Format triples into a readable string representation for prompt curation."""
    lines = ["# Triples (head:type, relation, tail:type):"]
    for triple in triples:
        head = triple.get("head", {})
        tail = triple.get("tail", {})
        relation = triple.get("relation", "")
        head_entity = head.get("entity", "")
        head_type = head.get("type", "")
        tail_entity = tail.get("entity", "")
        tail_type = tail.get("type", "")
        lines.append(
            f'("{head_entity}":{head_type}, "{relation}", "{tail_entity}":{tail_type})'
        )
    return "\n".join(lines)


def annotate_entities(text: str, prompt: str, iter: int = 1, threshold: float = 0.5) -> List[Dict]:
    """
    Annotate entities in the given text using an LLM with Self-Consistency.

    Args:
        text (str): The input text to annotate.
        prompt (str): The prompt to guide the annotation process.
        iter (int): Number of iterations for Self-Consistency. Defaults to 1 (no Self-Consistency).
        threshold (float): Fraction of iterations an entity must appear in to be kept.
                        Value between 0 and 1. Defaults to 0.5 (majority vote).

    Returns:
        List[Dict]: A list of dictionaries containing annotated entities:
            - "entity": str, the entity mention
            - "type": str, the entity type
    Notes:
        The function prefers a local vLLM backend and automatically falls back to
        an OpenAI-compatible API when vLLM cannot be used. Set `OPENAI_API_KEY`
        (or `LLM_API_KEY`) and optionally `OPENAI_BASE_URL`/`OPENAI_MODEL` for the
        API fallback.
    """

    def parse_entities_from_response(response: str) -> Set[Tuple[str, str]]:
        """
        Parse entities from the model response text.

        Args:
            response (str): Model output text in format "(entity | TYPE), ..."

        Returns:
            Set[Tuple[str, str]]: Set of extracted entity tuples (entity, type)
        """
        pattern = r"\(([^)]+)\s*\|\s*([^)]+)\)"
        matches = re.findall(pattern, response)
        return set((ent.strip(), typ.strip()) for ent, typ in matches)

    temperature = 0.8
    max_tokens = 512
    sampling_params = None
    if SamplingParams is not None:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1,
            top_k=-1,
            repetition_penalty=1.1,
            max_tokens=max_tokens,
            min_tokens=64
        )

    all_entities = []
    vllm_model = load_vllm(os.getenv("VLLM_MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta"))
    use_vllm = vllm_model is not None and sampling_params is not None
    backend_error: Optional[Exception] = None

    for _ in range(iter):
        response: Optional[str] = None

        if use_vllm:
            try:
                response = chat_vllm(
                    model=vllm_model,
                    system_query=prompt,
                    query=text,
                    sampling_params=sampling_params
                )
            except Exception as exc:  # pragma: no cover - propagate to API fallback
                logger.warning(
                    "vLLM backend unavailable, falling back to API mode: %s", exc
                )
                backend_error = exc
                use_vllm = False

        if response is None:
            try:
                response = chat_api(
                    query=text,
                    system_query=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                if backend_error is not None:
                    raise RuntimeError(
                        "Both vLLM and API backends failed"
                    ) from exc
                raise

        parsed_entities = parse_entities_from_response(response)
        all_entities.append(parsed_entities)

    # Count occurrences of each entity across iterations
    counter = Counter()
    for entity_set in all_entities:
        counter.update(entity_set)

    # Keep entities that appear in >= threshold fraction of iterations
    result = [
        {"entity": ent, "type": typ}
        for (ent, typ), count in counter.items()
        if count / iter >= threshold
    ]

    return result

def gen_entity_annotation_prompt(entities: List[Dict] = []) -> str:
    """
    Generate a prompt for entity annotation based on the provided entities.
    If no entities are given, fall back to two static examples with longer texts.

    Args:
        entities (List[Dict]): 
            A list of dictionaries, each containing:
                - "text" (str): The input text.
                - "entities" (list[str]): The extracted entities.

    Returns:
        str: A prompt for entity annotation with n-shot examples.
    """
    prompt = (
        "You are a text annotator. Your objectives are to:\n"
        "1/ Analyze the given text in detail\n"
        "2/ Annotate possible entities based on these entity types: "
        "person(PER), location(LOC), organization(ORG), time(TIME), numbers(NUM), "
        "and miscellaneous entity names(MISC).\n"
        'Respond in the format:\n"Possible Entities: e_1, e_2, ..., e_n"\n\n'
    )

    if entities:
        for entity in entities:
            text = entity.get("text", "")
            extracted = ", ".join(entity.get("entities", []))
            prompt += (
                "===\n"
                f"Text: {text}\n\n"
                f"Possible Extractions: {extracted}\n"
                "===\n\n"
            )
    else:
        # Two longer static examples as continuous paragraphs
        static_examples = [
            {
                "text": (
                    "Barack Obama, the 44th President of the United States, was born in Honolulu, Hawaii in 1961 and later studied at Columbia University in New York before earning a law degree from Harvard. In 2008 he was elected President after defeating John McCain, representing the Democratic Party, and during his presidency he signed the Affordable Care Act into law while also ordering the operation that killed Osama bin Laden in 2011."
                ),
                "entities": [
                    "(Barack Obama | PER)", "(United States | LOC)", "(Honolulu | LOC)", "(Hawaii | LOC)", 
                    "(1961 | TIME)", "(Columbia University | ORG)", "(New York | LOC)", 
                    "(Harvard | ORG)", "(2008 | TIME)", "(John McCain | PER)", 
                    "(Democratic Party | ORG)", "(Affordable Care Act | MISC)", 
                    "(Osama bin Laden | PER)", "(2011 | TIME)"
                ],
            },
            {
                "text": (
                    "Apple Inc., founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne, is headquartered in Cupertino, California and has become one of the world’s most valuable companies, best known for products such as the iPhone, iPad, and MacBook. The introduction of the iPhone in 2007 revolutionized the smartphone industry and under the leadership of Tim Cook the company has continued to expand globally with flagship stores and operations across Europe, Asia, and beyond."
                ),
                "entities": [
                    "(Apple Inc. | ORG)", "(1976 | TIME)", "(Steve Jobs | PER)", "(Steve Wozniak | PER)", 
                    "(Ronald Wayne | PER)", "(Cupertino | LOC)", "(California | LOC)", 
                    "(iPhone | MISC)", "(iPad | MISC)", "(MacBook | MISC)", 
                    "(2007 | TIME)", "(Tim Cook | PER)", "(Europe | LOC)", "(Asia | LOC)"
                ],
            },
        ]
        for example in static_examples:
            text = example["text"]
            extracted = ", ".join(example["entities"])
            prompt += (
                "===\n"
                f"Text: {text}\n\n"
                f"Possible Extractions: {extracted}\n"
                "===\n\n"
            )

    return prompt + "===\nText: "

def get_reference_kg(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a reference knowledge graph for faithfulness scoring."""
    return [{k:d[k] for k in list(d.keys())[:2]} for d in triples]


# This is the function to get K'
def naive_mapping(text: str, triples: List[Dict]) -> Tuple[List[Dict], Set[Tuple[str, str]]]:
    """
    Perform a naive mapping of triples/KG against a given text.
    
    - Keeps only triples where both head and tail entities appear in the text.
    - Tracks which entities were found in the text.
    
    Args:
        text (str): The input text to check against.
        triples (List[Dict]): A list of triples, where each triple contains:
            - "head" (dict): { "entity": str, "type": str }
            - "tail" (dict): { "entity": str, "type": str }
    
    Returns:
        Tuple[List[Dict], Set[Tuple[str, str]]]:
            - remaining_triples: Triples where both head and tail entities were found in the text.
            - used_entities: A set of entities (entity name, type) that appeared in the text.
    """
    entities: List[List[str]] = []
    filtered: List[Dict] = []
    used_entities: Set[Tuple[str, str]] = set()

    for triple in triples:
        head = triple["head"]["entity"]
        head_type = triple["head"]["type"]
        tail = triple["tail"]["entity"]
        tail_type = triple["tail"]["type"]

        # Track which entities appear in the text
        if re.search(r"\b" + re.escape(head) + r"\b", text):
            used_entities.add((head, head_type))
        if re.search(r"\b" + re.escape(tail) + r"\b", text):
            used_entities.add((tail, tail_type))

        entities.append([head, tail])
        filtered.append(triple)

    remaining_triples: List[Dict] = []
    for entity_list, triple in zip(entities, filtered):
        all_in_text = all(
            re.search(r"\b" + re.escape(entity) + r"\b", text) for entity in entity_list
        )
        if all_in_text:
            remaining_triples.append(triple)

    return remaining_triples, used_entities
