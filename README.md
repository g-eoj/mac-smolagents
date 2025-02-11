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

Grammar allows you to force structured output from the model using a logits processor.
You may want to use it for models that don't follow instructions well, to avoid errors or unnecessary token consumption (potentially reducing processing time and compute costs).

To use it, the model must be initialized with a logits processor.
Here we select the regex logits processor:

```python
mlx_language_model = mac_smolagents.MLXLModel(
    model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    logits_processor=mac_smolagents.logits_processors.RegexLogitsProcessor,
)
```

`mac-smolagents` provides grammars that attempt to match the output expected by `smolagents`.
You can also write your own grammar (it is just a string).
Here we are using a `CodeAgent` so we select the grammar that works for both the regex logits processor and `CodeAgent`:

```python
agent = smolagents.CodeAgent(
    model=mlx_language_model, 
    grammar=mac_smolagents.grammars.CodeAgentRegex,
    tools=[], 
    add_base_tools=True
)
```
Now we can run the agent as we normally would:

```python
agent.run(
    task="What happend on Jan 6th 2021?"
)
```
