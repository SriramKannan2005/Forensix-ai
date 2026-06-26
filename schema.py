from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ForensixResult:
    module: str
    file: str
    processing_time: float
    authenticity_score: float
    signals: dict
    flags: list
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def make_error_result(module: str, file: str, error_msg: str) -> dict:
    return ForensixResult(
        module=module,
        file=file,
        processing_time=0.0,
        authenticity_score=-1.0,
        signals={},
        flags=["ERROR"],
        error=error_msg
    ).to_dict()