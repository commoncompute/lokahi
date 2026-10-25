"""Bounded MLX compiler candidate plans; original model/state always restored.

The Python binding selects a native MLX compiler lowering; tensor math stays
in MLX/Metal. This does not claim independently written Metal or ANE execution.
"""
from contextlib import contextmanager
from dataclasses import dataclass

MODES = ('control', 'compile_decode', 'compile_prefill', 'compile_both')


@dataclass
class PreparedPlan:
    model: object
    mode: str
    originals: list
    wrappers: list
    stats: dict


def prepare(model, mode):
    """Keep compiled function identities alive across identical warm/measured calls."""
    if mode not in MODES:
        raise ValueError('candidate is outside the frozen execution-plan family')
    stats = {'candidate_calls': 0, 'fallback_calls': 0, 'installed_layers': 0}
    if mode == 'control':
        return PreparedPlan(model, mode, [], [], stats)
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import llama, qwen3
    if type(model) not in (llama.Model, qwen3.Model):
        raise ValueError('only exact dense Llama/Qwen3 models are eligible')
    layers = model.model.layers
    if not 1 <= len(layers) <= 64:
        raise ValueError('model layer bound exceeded')
    originals = [(layer, layer.mlp) for layer in layers]
    for layer, mlp in originals:
        if type(mlp) not in (llama.MLP, qwen3.MLP):
            raise ValueError('unsupported MLP semantics')
        for projection in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            if type(projection) is not nn.QuantizedLinear or (projection.bits, projection.group_size, projection.mode) != (4, 64, 'affine'):
                raise ValueError('only unchanged affine Q4 group64 is eligible')

    class CompiledMLP(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
            self._compiled = mx.compile(lambda x: original(x))

        def __call__(self, x):
            prefill = len(x.shape) == 3 and x.shape[0] == 1 and 1 < x.shape[1] <= 512
            decode = len(x.shape) == 3 and x.shape[0] == 1 and x.shape[1] == 1
            if ((prefill and mode in ('compile_prefill', 'compile_both')) or
                    (decode and mode in ('compile_decode', 'compile_both'))):
                stats['candidate_calls'] += 1
                return self._compiled(x)
            stats['fallback_calls'] += 1
            return self.original(x)

    return PreparedPlan(model, mode, originals,
                        [CompiledMLP(original) for _, original in originals], stats)


@contextmanager
def candidate(model, mode, *, prepared=None):
    plan = prepared if prepared is not None else prepare(model, mode)
    if plan.model is not model or plan.mode != mode:
        raise ValueError('prepared plan identity mismatch')
    if any(layer.mlp is not original for layer, original in plan.originals):
        raise ValueError('nested or changed candidate model')
    plan.stats.update(candidate_calls=0, fallback_calls=0, installed_layers=0)
    try:
        for (layer, _), wrapper in zip(plan.originals, plan.wrappers):
            layer.mlp = wrapper
            plan.stats['installed_layers'] += 1
        # Copy-on-yield is not possible: wrappers increment this dictionary.
        # Callers copy the completed counters into their immutable records.
        yield plan.stats
    finally:
        for layer, original in plan.originals:
            layer.mlp = original
