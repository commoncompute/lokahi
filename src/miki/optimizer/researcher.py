"""Replaceable reasoning provider; no inference/runtime network dependency."""
from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from ..runtime import MODES
from .integrity import CANDIDATE_FILE

SCHEMA = Path(__file__).with_name('proposal.schema.json')
FIELDS = {'title', 'mechanism', 'affected_code', 'expected_benefit', 'downside',
          'conditions', 'experiment', 'mode', 'discard', 'critique'}


def validate_proposals(value):
    if not isinstance(value, dict) or set(value) != {'hypotheses', 'evidence_limits'}:
        raise ValueError('invalid research contract')
    if not isinstance(value['hypotheses'], list) or not 1 <= len(value['hypotheses']) <= 3:
        raise ValueError('research population exceeds protected bounds')
    seen = set()
    for hypothesis in value['hypotheses']:
        if not isinstance(hypothesis, dict) or set(hypothesis) != FIELDS:
            raise ValueError('incomplete hypothesis')
        if hypothesis['mode'] not in MODES[1:] or hypothesis['mode'] in seen:
            raise ValueError('unbounded/duplicate execution-plan candidate')
        seen.add(hypothesis['mode'])
        if hypothesis['affected_code'] != CANDIDATE_FILE or type(hypothesis['discard']) is not bool:
            raise ValueError('candidate cannot alter protected implementation/evaluation files')
        for key in FIELDS - {'discard'}:
            if not isinstance(hypothesis[key], str) or not 1 <= len(hypothesis[key]) <= 4000:
                raise ValueError('hypothesis text is empty or unbounded')
    if (not isinstance(value['evidence_limits'], list) or len(value['evidence_limits']) > 20 or
            any(not isinstance(v, str) or len(v) > 4000 for v in value['evidence_limits'])):
        raise ValueError('invalid research evidence limitations')
    return value


class ResearchModel(ABC):
    @abstractmethod
    def request(self, stage, context):
        raise NotImplementedError

    def analyze_profile(self, context):
        return self.request('analyze_profile', context)

    def generate_hypotheses(self, context):
        return self.request('generate_hypotheses', context)

    def critique(self, context):
        return self.request('independent_adversarial_critique', context)

    def generate_candidate(self, hypothesis):
        # This first search family is a native compiler execution-plan search,
        # not permission to execute arbitrary LLM-generated source or commands.
        if hypothesis['mode'] not in MODES[1:] or hypothesis['affected_code'] != CANDIDATE_FILE:
            raise ValueError('invalid candidate scope')
        return {'schema_version': 1, 'mode': hypothesis['mode']}

    def interpret_result(self, context):
        return self.request('interpret_preserved_results_no_judging', context)


class CodexResearchModel(ResearchModel):
    def __init__(self, *, model='gpt-6-astra', timeout=180):
        self.model, self.timeout = model, timeout
        self.executable = shutil.which('codex')
        if not self.executable:
            raise RuntimeError('PROPOSER_UNAVAILABLE: Codex CLI is not installed')
        self.last_evidence = {}

    def request(self, stage, context):
        encoded = json.dumps(context, sort_keys=True, allow_nan=False)
        if len(encoded) > 60000:
            raise ValueError('research context exceeds bounded public-source envelope')
        prompt = (
            'You are the Miki research proposer, never its judge. Use only this supplied '
            'public source/physical-hardware evidence. Do not use tools, read files, run '
            'commands, edit anything, or invent measured wins. Return JSON matching the schema. '
            'The initial bounded family changes only miki_candidate.json: compile_decode, '
            'compile_prefill, compile_both. These invoke the native MLX compiler over '
            'unchanged exact dense Q4 MLPs, not new handcrafted Metal/ANE kernels. '
            'Give falsifiable mechanisms, downsides and experiments, not cosmetic variants. '
            'The critique is a separate independent reasoning pass: conservatively discard '
            'unsound/overfit proposals, including hidden weight copies, compiler changes, '
            'synchronization, numerical risks and invalid attribution. Missing profile '
            'evidence must remain missing. Acceptance, protected files, statistics, '
            'memory limits and workloads are immutable. Stage: ' + stage + '\n' + encoded)
        # Dedicated empty read-only workspace; no project, customer or model
        # files are exposed via cwd. Global configuration/hooks are not loaded.
        with tempfile.TemporaryDirectory(prefix='miki-research-') as folder:
            command = [self.executable, 'exec', '--ignore-user-config', '--ephemeral',
                       '--sandbox', 'read-only', '--skip-git-repo-check', '--cd', folder,
                       '--model', self.model, '--output-schema', str(SCHEMA), '--json', '-']
            result = subprocess.run(command, input=prompt, capture_output=True, text=True,
                                    timeout=self.timeout)
        events = []
        for line in result.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.last_evidence = {'provider': 'codex_cli', 'model': self.model,
                              'stage': stage, 'context_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
                              'returncode': result.returncode,
                              'usage': [e.get('usage') for e in events if e.get('usage')],
                              'llm_cost_currency': None}
        if any(e.get('type') == 'item.completed' and e.get('item', {}).get('type') in
               ('command_execution', 'file_change', 'mcp_tool_call', 'web_search') for e in events):
            raise ValueError('PROPOSER_CONTRACT_VIOLATION: research provider used a forbidden tool')
        if result.returncode:
            raise RuntimeError('PROPOSER_UNAVAILABLE: configured research request failed: ' + result.stderr[-2000:])
        messages = [e['item']['text'] for e in events if e.get('type') == 'item.completed'
                    and e.get('item', {}).get('type') == 'agent_message']
        if not messages:
            raise ValueError('research provider did not emit a structured final response')
        return validate_proposals(json.loads(messages[-1]))
