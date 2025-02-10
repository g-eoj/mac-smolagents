# mac-smolagents

This library allows you to use language models running on Apple silicon with [`smolagents`](https://github.com/huggingface/smolagents).

## Features

- Load and run models with [`mlx-lm`](https://pypi.org/project/mlx-lm/) (checkout [mlx-community](https://huggingface.co/mlx-community) for models to use).
- Enforce structured output from models using [grammar and `outlines`](https://dottxt-ai.github.io/outlines/latest/reference/generation/cfg/).

## Installation

`whl` files can be found under [releases](https://github.com/g-eoj/mac-smolagents/releases).

## Examples

### Quickstart

```python
import mac_smolagents
import smolagents


mlx_language_model = mac_smolagents.MLXLModel(
    model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"
)
agent = smolagents.CodeAgent(
    model=mlx_language_model, tools=[], add_base_tools=True
)
agent.run(
    task="What happend on Jan 6th 2021?"
)
```

### Using Grammar

```python
import mac_smolagents
import smolagents


mlx_language_model = mac_smolagents.MLXLModel(
    model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
)
agent = smolagents.CodeAgent(
    model=mlx_language_model, 
    grammar=mac_smolagents.CodeAgentGrammar,
    tools=[], 
    add_base_tools=True
)
agent.run(
    task="What happend on Jan 6th 2021?"
)
```
