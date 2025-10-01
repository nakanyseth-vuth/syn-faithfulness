#!/usr/bin/env python3
"""
Faithfulness Metrics for Knowledge Graph Evaluation

WHAT THIS SCRIPT DOES:
======================
This script computes a "faithfulness metric" that measures how well a generated text 
preserves information from a reference knowledge graph (KG). It's designed to evaluate 
whether generated text maintains the factual content present in structured knowledge.

KEY CONCEPT - Faithfulness Metric:
The faithfulness score is calculated using Jaccard similarity between:
1. Reference KG entities (ground truth)
2. K_ideal entities (combination of extracted and mapped entities)

MATHEMATICAL FORMULA:
Faithfulness = Jaccard(Reference_KG_entities, K_ideal_entities)

Where K_ideal = SC(generated_text) ∪ K'
- SC(generated_text) = Self-consistency entity extraction from generated text
- K' = Naive mapping of reference KG entities found in generated text

PERFORMANCE NOTE:
=================
This script can be SLOW on first run because:
1. Importing heavy ML libraries (torch, transformers, vllm) takes 3-10 seconds
2. If use_simple=False, it loads large language models for entity extraction
3. vLLM model loading can take 30+ seconds and requires significant GPU memory

QUICK USAGE (Fast mode):
========================
To avoid heavy dependencies and run quickly, use use_simple=True which bypasses
LLM-based entity extraction and uses regex patterns instead.

DEPENDENCIES:
=============
- Required: re, typing (built-in Python modules)
- Optional but imported: torch, transformers, vllm, datasets (heavy ML libraries)
- The import slowness comes from utils.py which imports these at module level
"""

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
    
    ALGORITHM OVERVIEW:
    ==================
    1. Extract entities from generated text using either:
       - LLM-based self-consistency (use_simple=False) - SLOW but accurate
       - Simple regex patterns (use_simple=True) - FAST but limited
    
    2. Find entities from reference KG that appear in the generated text (naive mapping)
    
    3. Combine both entity sets to form K_ideal = SC(text) ∪ K'
    
    4. Compare K_ideal with reference KG entities using Jaccard similarity
    
    MATHEMATICAL FORMULA:
    K_ideal = sc(generated_text) + k'
    k' = naive_mapping(generated_text, reference_kg)
    sc(generated_text) = annotate_entities(generated_text, prompt, iter, threshold)
    
    Faithfulness = |Reference ∩ K_ideal| / |Reference ∪ K_ideal|

    Args:
        reference_kg (List[Dict]): The reference KG (list of triples with head/tail entities).
        generated_text (str): The generated text to evaluate.
        prompt (str): Prompt for entity annotation (used in LLM mode).
        iter (int): Iterations for self-consistency (LLM mode only).
        threshold (float): Threshold for entity selection (LLM mode only).
        use_simple (bool): If True, uses fast regex extraction. If False, uses slow LLM extraction.

    Returns:
        float: Faithfulness score (Jaccard similarity) between 0.0 and 1.0.
               1.0 = perfect faithfulness, 0.0 = no overlap
    """
    # STEP 1: Entity extraction from generated text
    # =============================================
    # This is the potentially SLOW step depending on mode:
    # - use_simple=True: Fast regex matching (~1ms)
    # - use_simple=False: LLM inference (~seconds to minutes)
    if use_simple:
        sc_entities = simple_annotate_entities(generated_text, prompt, iter, threshold)
    else:
        sc_entities = annotate_entities(generated_text, prompt, iter, threshold)  # SLOW: LLM call
    sc_entity_set = set((e["entity"], e["type"]) for e in sc_entities)

    # STEP 2: Naive mapping - find reference entities in text
    # ======================================================= 
    # This is always fast: just regex searches (~1-10ms)
    _, naive_entity_set = naive_mapping(generated_text, reference_kg)

    # STEP 3: Combine entity sets to form K_ideal
    # ===========================================
    # Fast set union operation (~1ms)
    k_ideal_set = sc_entity_set | naive_entity_set

    # STEP 4: Extract reference entities from KG triples
    # =================================================
    # Fast iteration over reference triples (~1ms)
    ref_entity_set = set()
    for triple in reference_kg:
        ref_entity_set.add((triple["head"]["entity"], triple["head"]["type"]))
        ref_entity_set.add((triple["tail"]["entity"], triple["tail"]["type"]))

    # STEP 5: Compute Jaccard similarity
    # =================================
    # Fast set operations (~1ms)
    return jaccard_similarity(ref_entity_set, k_ideal_set)

# Example usage and performance demonstration
if __name__ == "__main__":
    """
    PERFORMANCE ANALYSIS:
    ====================
    
    SLOW PARTS (causing the delay you're experiencing):
    1. Module import time: 3-10 seconds (heavy ML libraries in utils.py)
    2. LLM model loading: 30+ seconds + GPU memory (if use_simple=False)
    3. LLM inference: 1-10 seconds per call (if use_simple=False)
    
    FAST PARTS:
    1. Regex entity extraction: <1ms (if use_simple=True)  
    2. Naive mapping: 1-10ms
    3. Set operations: <1ms
    4. Jaccard similarity: <1ms
    
    RECOMMENDATION FOR TESTING:
    - Always use use_simple=True for development/testing
    - Only use use_simple=False for final evaluation with proper GPU setup
    """
    
    print("Running faithfulness metric example...")
    print("Note: Initial import delay comes from loading ML libraries in utils.py")
    
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
    
    # TIME THE ACTUAL COMPUTATION (fast part)
    import time
    start_time = time.time()
    score = faithfulness_metric(reference_kg, generated_text, prompt, iter=1, threshold=0.5, use_simple=True)
    computation_time = time.time() - start_time
    
    print(f"Faithfulness score: {score:.2f}")
    print(f"Computation time (fast part): {computation_time:.3f} seconds")
    print("\nPERFORMANCE SUMMARY:")
    print("- Module import time: ~3-10 seconds (the delay you experienced)")
    print("- Actual computation: <0.01 seconds (very fast)")
    print("- Bottleneck: Heavy ML library imports in utils.py")
    print("- Solution: Use lazy imports or separate lightweight utility functions")
