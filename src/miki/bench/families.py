"""Pinned small-family canary plan, not runtime support or scaling evidence.

CommonCompute rows were read locally on 2026-10-05; this is a dated snapshot,
not a mutation of its catalog. Small distills/OCR do not cover native MoE/MLA.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

from ..system import real_path

GIB = 1024**3
EVOLVE_MODELS = frozenset(('Qwen3-0.6B-4bit', 'Llama-3.2-3B-Instruct-4bit'))


@dataclass(frozen=True)
class Canary:
    key: str
    family: str
    local_name: str
    repository: str
    revision: str
    architecture: str
    memory_tier_gib: int
    catalog_id: str | None
    scope: str
    does_not_cover: tuple[str, ...]
    verified_remote_weight_bytes: int | None = None


CANARIES = (
    Canary('qwen3-dense', 'qwen', 'Qwen3-0.6B-4bit', 'mlx-community/Qwen3-0.6B-4bit',
           '73e3e38d981303bc594367cd910ea6eb48349da8', 'qwen3', 8, 'qwen3-0.6b',
           'dense GQA/SwiGLU text control', ('hybrid recurrent state', 'MoE', 'vision')),
    Canary('qwen-hybrid', 'qwen', 'Qwen3.5-0.8B-OptiQ-4bit', 'mlx-community/Qwen3.5-0.8B-OptiQ-4bit',
           'ef60586933bd2cc02b763f77eb8839a5114bbec1', 'qwen3_5', 8, None,
           'research canary: hybrid recurrent text state; vision separate', ('MoE', 'Qwen3.8 shapes', 'VLM quality')),
    Canary('qwen38-fleet', 'qwen', 'Qwen3.8-27B-4bit', 'mlx-community/Qwen3.8-27B-4bit',
           '3e6447f082e89cc7f0bc6e5441afd38dfce760ff', 'qwen3_5', 24, 'qwen3.8-27b',
           'smallest general-purpose Qwen3.8 open checkpoint found; catalog weight estimate is wrong',
           ('resident 16 GiB M1 performance', 'full-size transfer from Qwen3.5 0.8B', 'MoE'),
           16054541349),
    Canary('llama-small', 'llama', 'Llama-3.2-1B-Instruct-4bit', 'mlx-community/Llama-3.2-1B-Instruct-4bit',
           '08231374eeacb049a0eade7922910865b8fce912', 'llama', 8, 'llama-3.2-1b',
           'small catalog dense GQA/SwiGLU text canary', ('Llama4 MoE', 'larger weight/KV footprints', 'vision')),
    Canary('llama-current', 'llama', 'Llama-3.2-3B-Instruct-4bit', 'mlx-community/Llama-3.2-3B-Instruct-4bit',
           '7f0dc925e0d0afb0322d96f9255cfddf2ba5636e', 'llama', 8, 'llama-3.2-3b',
           'already downloaded larger dense control', ('Llama4 MoE', '70B hardware performance', 'vision')),
    Canary('gemma3-text', 'gemma', 'gemma-3-1b-it-qat-4bit', 'mlx-community/gemma-3-1b-it-qat-4bit',
           '15fed4eafb456c6fcb2a1165f19ac609670ed14b', 'gemma3_text', 8, 'gemma-3-1b',
           'small catalog Gemma3 text canary, not the absolute smallest Gemma release',
           ('Gemma4 per-layer embeddings', 'MoE', 'multimodal encoders')),
    Canary('gemma4-small', 'gemma', 'gemma-4-e2b-it-4bit', 'mlx-community/gemma-4-e2b-it-4bit',
           '238767527555cb75a05732a84dff5d6ba0dd6809', 'gemma4', 8, 'gemma-4-e2b',
           'current catalog dense Gemma4 with per-layer embeddings; E2B is effective, not total parameters',
           ('Gemma4 26B MoE', 'Gemma4 12B encoder-free path', 'vision/audio qualification'),
           3550670554),
    Canary('deepseek-distill', 'deepseek', 'DeepSeek-R1-Distill-Qwen-1.5B-4bit',
           'mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit',
           '933185be1b8f81d9a21dcfa15ff73470d3545240', 'qwen2', 8, None,
           'research-only R1 distill, Qwen2 architecture; not a native DeepSeek canary',
           ('DeepSeek MLA', 'DeepSeek MoE/expert paging', 'DeepSeek V4 sparse attention')),
    Canary('deepseek-v4-fleet', 'deepseek', 'DeepSeek-V4-Flash', 'deepseek-ai/DeepSeek-V4-Flash',
           '60d8d70770c6776ff598c94bb586a859a38244f1', 'deepseek_v4', 256, None,
           '284B total / 13B active; research capacity tier only, native architecture unsupported locally',
           ('resident M1 execution', 'V4 CSA/HCA/mHC correctness from R1 distills', 'production support'),
           159617149040),
    Canary('glm-ocr', 'glm', 'GLM-OCR-4bit', 'mlx-community/GLM-OCR-4bit',
           '97f587506984cc92fa69b2694b4128e53db6b081', 'glm_ocr', 16, 'glm-ocr',
           'small OCR/VLM path only, requires image fixtures and VLM oracle',
           ('GLM chat', 'GLM4.7 MoE/MLA', 'GLM5 DSA')),
    Canary('glm-chat-fleet', 'glm', 'GLM-4.7-Flash-4bit', 'mlx-community/GLM-4.7-Flash-4bit',
           '1454cffb1a21737e162f508e5bc70be9def89276', 'glm4_moe_lite', 32, 'glm-4.7-flash',
           'smallest current catalog GLM chat lane; fleet or bounded native component fixtures on M1',
           ('GLM5 DSA', 'large-shape hardware performance')),
    Canary('glm5-latest-fleet', 'glm', 'GLM-5.3-Flash', 'zai-org/GLM-5.3-Flash',
           'eb9eb208eb0d988989d07a6a12d0fdeb5f52574a', 'glm5_next', 512, None,
           'current native GLM5 Flash: 320B total / 18B active, sparse/linear attention and mHC; unqualified research tier',
           ('resident M1 execution', 'native GLM5 parity from GLM-OCR or GLM4.7', 'production support'),
           328337455672),
    Canary('gpt-oss-fleet', 'gpt-oss', 'gpt-oss-20b-MXFP4-Q8', 'mlx-community/gpt-oss-20b-MXFP4-Q8',
           '773a7da77e569019bb0fd17a554b263738d669a3', 'gpt_oss', 32, 'gpt-oss-20b',
           'smallest GPT-OSS release; preserve MoE/MXFP4/Harmony, fleet target on this M1',
           ('120B hardware performance', 'MXFP4 parity from affine-Q4 tests')),
)

# Additional unchanged-MLX controls only. This does not admit new optimizer plans.
BASELINE_MODELS = EVOLVE_MODELS | frozenset(
    c.local_name for c in CANARIES if c.key in ('llama-small', 'gemma4-small', 'deepseek-distill'))


def plan(memory_bytes, installed_names=()):
    """Nominal catalog tier filter only: never a live memory/SSD admission bypass."""
    if type(memory_bytes) is not int or not 0 < memory_bytes < 2**64:
        raise ValueError('exact positive machine memory bytes required')
    installed = set(installed_names)
    rows = []
    for canary in CANARIES:
        tier_fits = memory_bytes >= canary.memory_tier_gib * GIB
        present = canary.local_name in installed
        status = ('FLEET_OR_COMPONENT_ONLY' if not tier_fits else
                  'LOCAL_SNAPSHOT_NEEDS_QUALIFICATION' if present else 'DOWNLOAD_DESTINATION_REQUIRED')
        rows.append({**asdict(canary), 'status': status, 'nominal_tier_fits': tier_fits,
                     'snapshot_present': present,
                     'memory_tier_source': 'CommonCompute catalog snapshot' if canary.catalog_id else 'unqualified research tier',
                     'controller_model_allowlisted': canary.local_name in EVOLVE_MODELS,
                     'artifact_and_live_admission_required': True,
                     'performance_qualified': False, 'fleet_scaling_verified': False})
    return {'schema_version': 1, 'snapshot_date': '2026-10-05', 'memory_bytes': memory_bytes,
            'scaling_policy': 'hypothesis_only_require_shape_model_quantization_state_and_chip_evidence',
            'download_performed': False, 'production_promotion_eligible': False, 'canaries': rows}


def local_plan(memory_bytes, model_root):
    root = real_path(Path(model_root))
    names = []
    for canary in CANARIES:
        child = root / canary.local_name
        # Resolve no symlinks, even for a present-but-unverified snapshot.
        real_path(child, exists=False)
        if child.is_dir():
            names.append(child.name)
    return plan(memory_bytes, names)
