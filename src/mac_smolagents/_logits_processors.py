import abc
import dataclasses
import mlx.core as mx
import mlx_lm
import outlines


class BaseLogitsProcessor(abc.ABC):
    @abc.abstractmethod
    def __init__(self, grammar: str, tokenizer: mlx_lm.tokenizer_utils.TokenizerWrapper):
        pass
    
    @abc.abstractmethod
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        pass


class CFGLogitsProcessor(BaseLogitsProcessor):
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


class RegexLogitsProcessor(BaseLogitsProcessor):
    def __init__(self, grammar, tokenizer):
        self._outlines_processor = outlines.processors.RegexLogitsProcessor(
            regex_string=grammar,
            tokenizer=outlines.models.TransformerTokenizer(tokenizer)
        )

    def __call__(self, input_ids, logits):
        processed_logits = self._outlines_processor(
            input_ids, 
            logits.reshape(-1)
        )
        return processed_logits.reshape(1, -1)

@dataclasses.dataclass()
class _LogitsProcessors:
    BaseLogitsProcessor=BaseLogitsProcessor
    CFGLogitsProcessor=CFGLogitsProcessor
    RegexLogitsProcessor=RegexLogitsProcessor
