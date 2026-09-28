"""Budget the actual JSON evidence block and record every selection decision."""

import json
from collections.abc import Iterable

from graphrag_bench.benchmark.context import ContextPiece, TokenCounter
from graphrag_bench.generation.config import GenerationError
from graphrag_bench.generation.contracts import TokenMeasurement
from graphrag_bench.models import text_sha256


def render_sources(pieces: Iterable[ContextPiece]) -> str:
    sources = [
        {"source_id": f"S{i}", "text": piece.text} for i, piece in enumerate(pieces, start=1)
    ]
    return json.dumps(sources, ensure_ascii=False, separators=(",", ":")) if sources else ""


class RecordingCounter:
    def __init__(self, provider: TokenCounter):
        self.provider = provider
        self.measurements: list[TokenMeasurement] = []

    def count_tokens(self, text: str) -> int:
        count = self.provider.count_tokens(text)
        if type(count) is not int or count < 1:
            raise GenerationError("tokenizer must return a positive integer for evidence")
        self.measurements.append(TokenMeasurement(text_sha256=text_sha256(text), count=count))
        return count


class ReplayCounter:
    """Replay recorded counts; this does not independently rerun the tokenizer."""

    def __init__(self, measurements: tuple[TokenMeasurement, ...]):
        self.remaining = iter(measurements)

    def count_tokens(self, text: str) -> int:
        measurement = next(self.remaining, None)
        if measurement is None or measurement.text_sha256 != text_sha256(text):
            raise GenerationError("token measurement does not match context candidate")
        return measurement.count

    def finish(self) -> None:
        if next(self.remaining, None) is not None:
            raise GenerationError("unused token measurements")
