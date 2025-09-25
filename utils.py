from typing import Tuple, List, Dict, Set, Any
from collections import Counter
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from vllm import LLM, SamplingParams




def chat_vllm(
    model,
    query: str,
    system_query: str,
    sampling_params: SamplingParams
):
    if system_query == '':
        messages=[
            {"role": "user", "content": query}
        ]
    else:
        messages=[
            {"role": "system", "content": system_query},
            {"role": "user", "content": query}
        ]
    output = model.chat(messages, sampling_params, use_tqdm=False)
    prompt = output[0].prompt
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


def load_vllm(model_name_or_path: str = "HuggingFaceH4/zephyr-7b-beta") -> LLM:
    """
    Load a vLLM model for text generation.

    Args:
        model_name_or_path (str, optional): The model name or path for vLLM. 
            Defaults to "HuggingFaceH4/zephyr-7b-beta"

    Returns:
        LLM: A vLLM model instance ready for text generation.
    """
    model = LLM(model=model_name_or_path, task="generate")
    return model


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

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=1,
        top_k=-1,
        repetition_penalty=1.1,
        max_tokens=512,
        min_tokens=64
    )

    all_entities = []

    for _ in range(iter):
        response = chat_vllm(
            model=load_vllm(),
            system_query=prompt,
            query=text,
            sampling_params=sampling_params
        )
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

