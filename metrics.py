from typing import List, Dict, Set, Tuple
from utils import naive_mapping, annotate_entities

# Fallback for annotate_entities for simple tests (no vllm required)
import re
def simple_annotate_entities(text: str, prompt: str, iter: int = 1, threshold: float = 0.5) -> list:
    """
    Simple entity extraction using regex for demonstration purposes.
    Extracts entities in the format (entity | TYPE) from the text if present.
    """
    pattern = r"\(([^)]+) \| ([^)]+)\)"
    matches = re.findall(pattern, text)
    return [{"entity": ent.strip(), "type": typ.strip()} for ent, typ in matches]

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Compute Jaccard similarity between two sets.
    """
    if not set1 and not set2:
        return 1.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 1.0


def faithfulness_metric(reference_kg: List[Dict], generated_text: str, prompt: str, iter: int = 1, threshold: float = 0.5, use_simple: bool = False) -> float:
    """
    Compute faithfulness metric between reference KG and generated KG (K_ideal) using Jaccard similarity.
    K_ideal = sc(generated_text) + k'
    k' = naive_mapping(generated_text, reference_kg)
    sc(generated_text) = annotate_entities(generated_text, prompt, iter, threshold)

    Args:
        reference_kg (List[Dict]): The reference KG (list of triples).
        generated_text (str): The generated text.
        prompt (str): Prompt for entity annotation.
        iter (int): Iterations for self-consistency.
        threshold (float): Threshold for entity selection.

    Returns:
        float: Faithfulness score (Jaccard similarity).
    """
    # Self-consistency entities from generated text
    if use_simple:
        sc_entities = simple_annotate_entities(generated_text, prompt, iter, threshold)
    else:
        sc_entities = annotate_entities(generated_text, prompt, iter, threshold)
    sc_entity_set = set((e["entity"], e["type"]) for e in sc_entities)

    # Naive mapping entities from reference KG
    _, naive_entity_set = naive_mapping(generated_text, reference_kg)

    # K_ideal: union of self-consistency and naive mapping entities
    k_ideal_set = sc_entity_set | naive_entity_set

    # Reference KG entities
    ref_entity_set = set()
    for triple in reference_kg:
        ref_entity_set.add((triple["head"]["entity"], triple["head"]["type"]))
        ref_entity_set.add((triple["tail"]["entity"], triple["tail"]["type"]))

    # Faithfulness metric: Jaccard similarity
    return jaccard_similarity(ref_entity_set, k_ideal_set)

# Example usage
if __name__ == "__main__":
    # Example input in the new format
    example = {
        "text": (
            "Apple Inc., founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne, is headquartered in Cupertino, California and has become one of the world’s most valuable companies, best known for products such as the iPhone, iPad, and MacBook. The introduction of the iPhone in 2007 revolutionized the smartphone industry and under the leadership of Tim Cook the company has continued to expand globally with flagship stores and operations across Europe, Asia, and beyond."
        ),
        "entities": [
            "(Apple Inc. | ORG)", "(1976 | TIME)", "(Steve Jobs | PER)", "(Steve Wozniak | PER)", 
            "(Ronald Wayne | PER)", "(Cupertino | LOC)", "(California | LOC)", 
            "(iPhone | MISC)", "(iPad | MISC)", "(MacBook | MISC)", 
            "(2007 | TIME)", "(Tim Cook | PER)", "(Europe | LOC)", "(Asia | LOC)"
        ],
    }

    # Convert entities to reference KG triples (for demonstration, pairwise as head-tail)
    def parse_entity(entity_str):
        match = re.match(r"\((.+) \| (.+)\)", entity_str)
        if match:
            return {"entity": match.group(1).strip(), "type": match.group(2).strip()}
        return None

    entities = [parse_entity(e) for e in example["entities"] if parse_entity(e)]
    # Create dummy triples: each entity as head, next as tail
    reference_kg = []
    for i in range(len(entities)-1):
        reference_kg.append({"head": entities[i], "tail": entities[i+1]})

    generated_text = example["text"]
    prompt = "Annotate all entities in the text."
    score = faithfulness_metric(reference_kg, generated_text, prompt, iter=1, threshold=0.5, use_simple=True)
    print(f"Faithfulness score: {score:.2f}")
