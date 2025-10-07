"""Utility for producing baseline synthetic data artifacts for KG evolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

# Resolve paths relative to this script so it works from any current directory.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "initial_synthetic_data.json"
DEFAULT_BASELINE = SCRIPT_DIR / "data" / "initial_synthetic_text.txt"

from utils import format_triples_for_input, load_redocred, chat_api, load_vllm, chat_vllm, format_dataset_redocred, get_reference_kg
from metrics_fast import faithfulness_metric


def write_dataset(examples: List[Dict[str, Any]], output_path: Path) -> None:
    """Write the list of examples as JSON."""
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(examples, handle, indent=2, ensure_ascii=False)


def write_baseline(example: Dict[str, Any], baseline_path: Path) -> None:
    """Write the baseline text file consumed by run_evolution.sh."""
    with baseline_path.open("w", encoding="utf-8") as handle:
        handle.write(example["synthetic_text"].strip() + "\n")


def resolve_path(path: Path) -> Path:
    """Resolve CLI paths relative to the script directory."""
    return path if path.is_absolute() else SCRIPT_DIR / path


def generate_initial_text_from_kg(triples, prompt=None, use_vllm=True):
    """
    Generate initial text from KG triples, prefer vllm, fallback to API if vllm fails.
    """
    # Format input
    reformatted_triples = format_triples_for_input(triples)
    input_text = f"""You are a creative english text writer. Write me a text using the provided Knowledge Graph. Your objective is to write a coherent text that incorporates all the given triples (head, relation, tail) of the Knowledge Graph. You have the right to make the text creative and informative, but you must make sure that the text reflects the given Knowledge Graph.

# Triples (head:type, relation, tail:type):
("Stockholm":LOC, "country", "Sweden":LOC)
("Stockholm":LOC, "country", "Swedish":LOC)
("Harpo":PER, "country of citizenship", "Sweden":LOC)
("Harpo":PER, "country of citizenship", "Swedish":LOC)
("Bergqvist":PER, "date of birth", "6 October 1962":TIME)
("Niklas Bergqvist":PER, "date of birth", "6 October 1962":TIME)
("Bergqvist":PER, "place of birth", "Stockholm":LOC)
("Niklas Bergqvist":PER, "place of birth", "Stockholm":LOC)
("Bergqvist":PER, "country of citizenship", "Sweden":LOC)
("Bergqvist":PER, "country of citizenship", "Swedish":LOC)
("Niklas Bergqvist":PER, "country of citizenship", "Sweden":LOC)
("Niklas Bergqvist":PER, "country of citizenship", "Swedish":LOC)

# Text:
Niklas Bergqvist (born 6 October 1962 in Stockholm), is a Swedish songwriter, producer and musician. His career started in 1984, as a founding member of the Synthpop band Shanghai, which he started along his school friends. The band released two albums and numerous singles, with their debut single "Ballerina", produced by former Europe guitarist Kee Marcello. Following the success of the single, the album also gained positive attention, resulting in the band touring around Sweden. The second, and final, studio album was produced by Swedish pop star Harpo. After the band split-up in 1987, Bergqvist formed and played in several other bands until he decided to focus more on songwriting, resulting in several releases ever since. In 2015, he founded the independent record label Tracks of Sweden with songwriting partner Simon Johansson.

{reformatted_triples}

# Text:
"""
    system_prompt = prompt or "You are a knowledge graph text generator. Generate a coherent paragraph covering all entities and relations in the triples."
    temperature = 0.7
    max_tokens = 1024
    sampling_params = None
    vllm_model = None
    if use_vllm:
        try:
            vllm_model = load_vllm()
            if vllm_model is not None:
                from utils import SamplingParams
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=1,
                    top_k=-1,
                    repetition_penalty=1.1,
                    max_tokens=max_tokens,
                    min_tokens=64
                )
                text = chat_vllm(
                    model=vllm_model,
                    query=input_text,
                    system_query=system_prompt,
                    sampling_params=sampling_params
                )
                return text
        except Exception as exc:
            print(f"[WARN] vLLM failed: {exc}")
    # API fallback
    try:
        text = chat_api(
            query=input_text,
            system_query=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return text
    except Exception as exc:
        print(f"[ERROR] API failed: {exc}")
        return ""


def main():
    # Example: load KG (replace with actual KG loading logic if needed)
    dataset = load_redocred()["train_annotated"]
    annotated = format_dataset_redocred(dataset)
    # Here only use the triples from the first document for demonstration
    triples = annotated[0]["triples"]
    synthetic_text = generate_initial_text_from_kg(triples)
    example = {
        "synthetic_text": synthetic_text,
        "triples": triples,
    }
    write_dataset([example], DEFAULT_OUTPUT)
    write_baseline(example, DEFAULT_BASELINE)
    print(f"Initial text and data generated: {DEFAULT_OUTPUT}, {DEFAULT_BASELINE}")

    reference_kg = get_reference_kg(triples)
    faithfulness_score = faithfulness_metric(reference_kg, synthetic_text)
    print(f"Faithfulness Score: {faithfulness_score:.4f}")


if __name__ == "__main__":
    main()

