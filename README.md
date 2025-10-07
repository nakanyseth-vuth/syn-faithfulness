# syn-faithfulness

A pipeline for evolving and evaluating synthetic text generated from Knowledge Graphs (KGs), built on top of the OpenEvolve evolutionary framework.

## Features

- **Initial Data Generation**: Uses LLMs (vLLM or OpenAI-compatible API) to generate baseline synthetic text from KG triples.
- **Faithfulness Evaluation**: Custom metrics to assess how well the synthetic text reflects the original KG.
- **Evolutionary Optimization**: Iteratively improves synthetic text using LLMs and prompt-based mutation, guided by faithfulness scores.
- **Configurable Pipeline**: All settings (LLM, prompt, evaluation, etc.) are controlled via `config.yaml`.
- **OpenEvolve Integration**: Leverages OpenEvolve for evolutionary orchestration, checkpointing, and parallelism.

## Project Structure

- `generate_initial_data.py` — Generate initial synthetic text and data from KG triples.
- `evaluator.py` — Evaluate the faithfulness of synthetic text to the KG.
- `run_evolution.py` — Main Python script to orchestrate the full evolution pipeline.
- `config.yaml` — All pipeline configuration (LLM, prompt, evaluation, etc).
- `data/` — Stores initial and evolved synthetic data.
- `templates/` — Prompt templates for LLM-based evolution.
- `openevolve/` — OpenEvolve framework and core evolutionary logic.

## Quick Start

1. Install dependencies (see OpenEvolve requirements).
2. Set your OpenAI-compatible API key:  
	`export OPENAI_API_KEY='your_api_key_here'`
3. Generate initial data:  
	`python generate_initial_data.py`
4. Run the evolution pipeline:  
	`python run_evolution.py 10`  
	(replace `10` with desired number of iterations)

## References

- Built on [OpenEvolve](https://github.com/codelion/openevolve) for evolutionary optimization.
- See `config.yaml` and `templates/` for customization.
