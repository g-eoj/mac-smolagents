import dataclasses
import json
import mlx_lm
import outlines
import smolagents
import uuid

from typing import Dict, List, Optional


class CFGLogitsProcessor:
    def __init__(self, grammar, tokenizer):
        self._outlines_processor = outlines.processors.CFGLogitsProcessor(
            cfg_str=grammar,
            tokenizer=outlines.models.TransformerTokenizer(tokenizer)
        )

    def __call__(self, input_ids, logits):
        processed_logits = self._outlines_processor(
            input_ids, 
            logits.reshape(-1)
        )
        return processed_logits.reshape(1, -1)


class MLXLModel(smolagents.Model):
    """A class to interact with models loaded using mlx-lm on Apple silicon.

    Parameters:
        model_id (str):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
        trust_remote_code (bool):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
        kwargs (dict, *optional*):
            Any additional keyword arguments that you want to use in model.generate(), for instance `max_tokens`.

    Example:
    ```python
    >>> engine = MLXLModel(
    ...     model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    ...     max_tokens=10000,
    ... )
    >>> messages = [
    ...     {
    ...         "role": "user", 
    ...         "content": [
    ...             {"type": "text", "text": "Explain quantum mechanics in simple terms."}
    ...         ]
    ...     }
    ... ]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.model, self.tokenizer = mlx_lm.load(model_id, tokenizer_config={"trust_remote_code": trust_remote_code})

    def _to_message(self, text, tools_to_call_from):
        if tools_to_call_from:
            maybe_json = "{" + text.split("{", 1)[-1][::-1].split("}", 1)[-1][::-1] + "}"
            parsed_text = json.loads(maybe_json)
            tool_name = parsed_text.get("name", None)
            tool_arguments = parsed_text.get("arguments", None)
            if tool_name:
                return smolagents.models.ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        smolagents.models.ChatMessageToolCall(
                            id=uuid.uuid1(),
                            type="function",
                            function=smolagents.models.ChatMessageToolCallDefinition(name=tool_name, arguments=tool_arguments),
                        )
                    ],
                )
        return smolagents.models.ChatMessage(role="assistant", content=text)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[smolagents.Tool]] = None,
        **kwargs,
    ) -> smolagents.ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            flatten_messages_as_text=True,  # mlx-lm doesn't support vision models
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        
        # completion_kwargs post-process steps needed for mlx-lm
        messages = completion_kwargs.pop("messages")
        prepared_stop_sequences = completion_kwargs.pop("stop", [])
        tools = completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)
        grammar = completion_kwargs.pop("grammar", None)
        if grammar:
            completion_kwargs["logits_processors"] = [
                CFGLogitsProcessor(grammar=grammar, tokenizer=self.tokenizer)
            ]
            
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
        )

        self.last_input_token_count = len(prompt_ids)
        self.last_output_token_count = 0
        text = ""

        for step in mlx_lm.stream_generate(self.model, self.tokenizer, prompt=prompt_ids, **completion_kwargs):
            #print(step.text, end="", flush=True)
            self.last_output_token_count += 1
            text += step.text
            for stop_sequence in prepared_stop_sequences + [self.tokenizer.eos_token]:
                if text.rstrip().endswith(stop_sequence):
                    text = text.rstrip()[: -len(stop_sequence)]
                    return self._to_message(text, tools_to_call_from)

        return self._to_message(text, tools_to_call_from)
