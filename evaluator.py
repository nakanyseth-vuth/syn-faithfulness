"""
Evaluator for Knowledge Graph synthetic data evolution.
Returns EvaluationResult objects as expected by OpenEvolve framework.
"""
from __future__ import annotations

import os
from pathlib import Path
import json
import traceback
from typing import List, Dict, Any

# Path to initial data
INITIAL_DATA_PATH = Path(__file__).resolve().parent / "data" / "initial_synthetic_data.json"

# Import OpenEvolve EvaluationResult
from openevolve.evaluation_result import EvaluationResult
from metrics_fast import faithfulness_metric

def load_initial_data() -> List[Dict[str, Any]]:
    """Load initial KG synthetic data examples."""
    if os.path.exists(INITIAL_DATA_PATH):
        with open(INITIAL_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

INITIAL_DATA = load_initial_data()



def get_reference_kg(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse knowledge graph triples from text format."""
    return [{k:d[k] for k in list(d.keys())[:2]} for d in triples]

ARTIFACTS = get_reference_kg(INITIAL_DATA[0]["triples"]) if INITIAL_DATA else []    
def evaluate_synthetic_text(synthetic_text: str, triples: List[Dict[str, Any]]) -> float:
    """Evaluate faithfulness of synthetic text to knowledge graph."""
    # Parse KG triples
    reference_kg = get_reference_kg(triples)
    
    if not reference_kg:
        print("Warning: No valid KG triples found")
        return 0.0
    
    # Calculate faithfulness score with correct parameter order
    score = faithfulness_metric(reference_kg, synthetic_text)
    return score


def calculate_data_features(synthetic_text: str, triples: List[Dict[str, Any]]) -> tuple:
    """
    Calculate custom features for MAP-Elites
    
    Returns:
        tuple: (text_length, entity_density)
        - text_length: Actual character count
        - entity_density: Continuous score 0.0-1.0
    """
    # Feature 1: Text length (raw character count)
    text_length = len(synthetic_text)
    
    # Feature 2: Entity density score (continuous 0.0-1.0)
    words = synthetic_text.split()
    if not words:
        return text_length, 0.0
    
    # Parse KG to get actual entities
    reference_kg = get_reference_kg(triples)
    if not reference_kg:
        # Fallback: count capitalized words as potential entities
        potential_entities = sum(1 for word in words if word[0].isupper() and len(word) > 2)
        entity_density = potential_entities / len(words)
    else:
        # Count actual entities from KG that appear in text
        all_entities = set()
        for triple in reference_kg:
            all_entities.add(triple["head"]["entity"])
            all_entities.add(triple["tail"]["entity"])
        
        entities_found = 0
        for entity in all_entities:
            if entity.lower() in synthetic_text.lower():
                entities_found += 1
        
        # Normalize by total entities
        entity_density = entities_found / len(all_entities) if all_entities else 0.0
    
    # Ensure density is within 0.0-1.0 range
    entity_density = min(1.0, max(0.0, entity_density))
    
    return text_length, entity_density


def load_program_data(program_path: str) -> Dict[str, Any]:
    """
    Load synthetic data from program file.
    
    Args:
        program_path: Path to the program file containing synthetic data
        
    Returns:
        Dictionary with synthetic_text and kg_triples
    """
    try:
        with open(program_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        initial_data = load_initial_data()
        if initial_data:
            # Use first example's synthetic text and KG triples
            return {
                "synthetic_text": content,
                "triples": initial_data[0]["triples"]
            }
        else:
            raise ValueError("No initial data available and program is not JSON")

    except Exception as e:
        print(f"Error loading program data: {e}")
        return None


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """
    Stage 1 evaluation: Quick evaluation of synthetic data.
    
    Args:
        program_path: Path to the program file containing synthetic data
        
    Returns:
        EvaluationResult with evaluation metrics
    """
    print("-" * 80)
    print("Starting Stage 1 evaluation (Synthetic Data Evolution)...")
    print("-" * 80)
    
    try:
        # Load program data
        program_data = load_program_data(program_path)
        if not program_data:
            metrics = {
                "faithfulness_score": 0.0,
                "text_length": 0,
                "entity_density": 0.0,
                "error": "Failed to load program data"
            }
            return EvaluationResult(metrics=metrics, artifacts=ARTIFACTS)
        
        synthetic_text = program_data.get("synthetic_text", "")
        kg_triples = program_data.get("triples", "")
        
        print(f"Loaded synthetic text ({len(synthetic_text)} characters)")
        
        # Evaluate faithfulness
        faithfulness_score = evaluate_synthetic_text(synthetic_text, kg_triples)
        print(f"Stage 1 faithfulness: {faithfulness_score:.3f}")
        
        # Calculate features
        text_length, entity_density = calculate_data_features(synthetic_text, kg_triples)
        print(f"Data features - Length: {text_length}, Entity density: {entity_density:.3f}")
        
        print("-" * 80)
        
        metrics = {
            "faithfulness_score": faithfulness_score,
            "text_length": text_length,
            "entity_density": entity_density,
            "combined_score": faithfulness_score,
            "program_artifacts": INITIAL_DATA[0]["triples"]
        }

        return EvaluationResult(metrics=metrics, artifacts=ARTIFACTS)

    except Exception as e:
        print(f"Stage 1 evaluation failed: {str(e)}")
        traceback.print_exc()
        print("-" * 80)
        
        # Always return feature dimensions, even on failure
        try:
            program_data = load_program_data(program_path)
            if program_data:
                synthetic_text = program_data.get("synthetic_text", "")
                kg_triples = program_data.get("triples", "")
                text_length, entity_density = calculate_data_features(synthetic_text, kg_triples)
            else:
                text_length, entity_density = 0, 0.0
        except:
            text_length, entity_density = 0, 0.0
        
        metrics = {
            "faithfulness_score": 0.0,
            "text_length": text_length,
            "entity_density": entity_density,
            "combined_score": 0.0,
            "program_artifacts": INITIAL_DATA[0]["triples"],
            "error": str(e)
        }

        return EvaluationResult(metrics=metrics, artifacts=ARTIFACTS)


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """
    Stage 2 evaluation: Comprehensive evaluation of synthetic data.
    
    Args:
        program_path: Path to the program file containing synthetic data
        
    Returns:
        EvaluationResult with evaluation metrics
    """
    print("-" * 80)
    print("Starting Stage 2 evaluation (Synthetic Data Evolution)...")
    print("-" * 80)
    
    try:
        # Load program data
        program_data = load_program_data(program_path)
        if not program_data:
            metrics = {
                "faithfulness_score": 0.0,
                "text_length": 0,
                "entity_density": 0.0,
                "combined_score": 0.0,
                "program_artifacts": INITIAL_DATA[0]["triples"],
                "error": "Failed to load program data"
            }
            return EvaluationResult(metrics=metrics, artifacts=ARTIFACTS)
        
        synthetic_text = program_data.get("synthetic_text", "")
        kg_triples = program_data.get("triples", "")
        
        print(f"Loaded synthetic text ({len(synthetic_text)} characters)")
        
        # Evaluate faithfulness with more detailed analysis
        faithfulness_score = evaluate_synthetic_text(synthetic_text, kg_triples)
        
        # Additional quality metrics for Stage 2
        reference_kg = get_reference_kg(kg_triples)
        
        # Count entity coverage
        entity_coverage = 0
        total_entities = 0
        entities_found = 0
        if reference_kg:
            all_entities = set()
            for triple in reference_kg:
                all_entities.add(triple["head"]["entity"])
                all_entities.add(triple["tail"]["entity"])
            
            total_entities = len(all_entities)
            for entity in all_entities:
                if entity.lower() in synthetic_text.lower():
                    entities_found += 1
            
            entity_coverage = entities_found / total_entities if total_entities > 0 else 0
        
        print(f"Stage 2 faithfulness: {faithfulness_score:.3f}")
        print(f"Entity coverage: {entity_coverage:.3f} ({entities_found}/{total_entities})")
        
        # Calculate features
        text_length, entity_density = calculate_data_features(synthetic_text, kg_triples)
        print(f"Data features - Length: {text_length}, Entity density: {entity_density:.3f}")
        
        print("-" * 80)
        
        metrics = {
            "faithfulness_score": faithfulness_score,
            "text_length": text_length,
            "entity_density": entity_density,
            "entity_coverage": entity_coverage,
            "total_entities": total_entities,
            "combined_score": faithfulness_score,
            "program_artifacts": INITIAL_DATA[0]["triples"],
        }

        return EvaluationResult(metrics=metrics, artifacts=ARTIFACTS)

    except Exception as e:
        print(f"Stage 2 evaluation failed: {str(e)}")
        traceback.print_exc()
        print("-" * 80)
        
        # Always return feature dimensions, even on failure
        try:
            program_data = load_program_data(program_path)
            if program_data:
                synthetic_text = program_data.get("synthetic_text", "")
                kg_triples = program_data.get("kg_triples", "")
                text_length, entity_density = calculate_data_features(synthetic_text, kg_triples)
            else:
                text_length, entity_density = 0, 0.0
        except:
            text_length, entity_density = 0, 0.0
        
        metrics = {
            "faithfulness_score": 0.0,
            "text_length": text_length,
            "entity_density": entity_density,
            "combined_score": 0.0,
            "program_artifacts": INITIAL_DATA[0]["triples"],
            "error": str(e)
        }

        return EvaluationResult(metrics=metrics, artifacts=ARTIFACTS)


def evaluate(program_path: str) -> EvaluationResult:
    """
    Main evaluation function - for backwards compatibility
    Calls evaluate_stage2 for full evaluation
    
    Args:
        program_path: Path to the program file containing synthetic data
        
    Returns:
        EvaluationResult with combined_score metric
    """
    return evaluate_stage2(program_path)


if __name__ == "__main__":
    # Test the evaluator
    print("Testing KG Synthetic Data Evaluator")
    print("=" * 50)
    
    # Check if initial data exists
    if os.path.exists(INITIAL_DATA_PATH):
        initial_data = load_initial_data()
        if initial_data:
            print(f"Found {len(initial_data)} initial data samples")
            
            # Test with first sample
            test_data = {
                "synthetic_text": initial_data[0]["synthetic_text"],
                "triples": initial_data[0]["triples"]
            }
            
            # Save as test program
            test_program_path = "initial_synthetic_text.txt"
            with open(test_program_path, "w", encoding="utf-8") as f:
                f.write(test_data["synthetic_text"])
            # Run evaluation
            result = evaluate_stage1(test_program_path)
            print("\nEvaluation Result:")
            for key, value in result.metrics.items():
                print(f"  {key}: {value}")
            
            # Cleanup
            os.remove(test_program_path)
        else:
            print("No initial data found")
    else:
        print(f"Initial data file not found at {INITIAL_DATA_PATH}")
        print("Please run generate_initial_data.py first")