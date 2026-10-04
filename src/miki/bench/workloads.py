"""Protected fixed-work workloads. Changes require a new policy/baseline."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Workload:
    name: str
    prompt_tokens: int
    output_tokens: int
    seed: str


WORKLOADS = (
    Workload('interactive', 64, 32, 'Explain how a language model processes a prompt. '),
    Workload('prefill', 256, 32, 'Describe a deterministic algorithm and its memory requirements. '),
)
WARMUPS = 3
INITIAL_PAIRS = 10
MAX_PAIRS = 30


def manifest():
    return [asdict(workload) for workload in WORKLOADS]


def prompt(tokenizer, workload):
    seed = [int(t) for t in tokenizer.encode(workload.seed, add_special_tokens=False)
            if t not in tokenizer.eos_token_ids]
    if not seed:
        raise ValueError('empty protected prompt')
    return (seed * ((workload.prompt_tokens + len(seed) - 1) // len(seed)))[:workload.prompt_tokens]
