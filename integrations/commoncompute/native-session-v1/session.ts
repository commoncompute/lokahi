/** Metadata-only diagnostic codec. No plaintext, transport, model execution,
 * remote admission authority or production qualification. All uint64 counters
 * are decimal strings: JS numbers must never silently round native bytes. */
export interface SessionDiagnostic {
  schema_version: 1;
  request_id: string;
  model_digest: string;
  reserved_bytes: string;
  revision: string;
  state: number;
  reason: number;
  prompt_tokens: number;
  completion_tokens: number;
  max_output_tokens: number;
  native_runtime_qualified: false;
}
const keys = ['schema_version', 'request_id', 'model_digest', 'reserved_bytes',
  'revision', 'state', 'reason', 'prompt_tokens', 'completion_tokens',
  'max_output_tokens', 'native_runtime_qualified'].sort();
function integer(value: unknown, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min || value > max) {
    throw new Error('invalid bounded native integer');
  }
  return value;
}
function fullMatch(pattern: RegExp, value: string): boolean {
  // JS $ may match before a final newline; canonical ABI strings may not.
  return pattern.exec(value)?.[0] === value;
}
function uint64(value: unknown, nonzero = false): string {
  if (typeof value !== 'string' || !fullMatch(/^(0|[1-9][0-9]{0,19})$/, value)
      || BigInt(value) > 18446744073709551615n || (nonzero && BigInt(value) === 0n)) {
    throw new Error('native uint64 must be an exact decimal string');
  }
  return value;
}
export function decodeSessionDiagnostic(value: unknown): SessionDiagnostic {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('invalid diagnostic');
  const v = value as Record<string, unknown>;
  if (JSON.stringify(Object.keys(v).sort()) !== JSON.stringify(keys)
      || v.schema_version !== 1 || v.native_runtime_qualified !== false
      || typeof v.request_id !== 'string' || !fullMatch(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,126}$/, v.request_id)
      || typeof v.model_digest !== 'string' || !fullMatch(/^[0-9a-f]{64}$/, v.model_digest)
      || /^0+$/.test(v.model_digest)) throw new Error('invalid identity/keys/qualification');
  const state = integer(v.state, 1, 6), reason = integer(v.reason, 0, 5);
  const prompt = integer(v.prompt_tokens, 1, 65535), max = integer(v.max_output_tokens, 1, 16384);
  const completion = integer(v.completion_tokens, 0, max);
  if (prompt + max > 65536) throw new Error('invalid context accounting');
  const reserve = uint64(v.reserved_bytes), revision = uint64(v.revision, true);
  if ((state <= 3) !== (BigInt(reserve) > 0n)
      || (state <= 2 && reason !== 0) || (state === 1 && completion !== 0)
      || ((state === 3 || state === 6) && ![4, 5].includes(reason))
      || (state === 4 && !(reason === 1 || (reason === 2 && completion === max)))
      || (state === 5 && reason !== 3)) throw new Error('invalid lifecycle/accounting');
  return {schema_version: 1, request_id: v.request_id, model_digest: v.model_digest,
    reserved_bytes: reserve, revision, state, reason, prompt_tokens: prompt,
    completion_tokens: completion, max_output_tokens: max, native_runtime_qualified: false};
}
/** Per-request observer only. Revision is cumulative snapshot state, not a
 * complete ordered text-event stream and not a durable billing receipt. */
export class SessionObservation {
  private last?: SessionDiagnostic;
  accept(value: unknown): boolean {
    const next = decodeSessionDiagnostic(value), last = this.last;
    if (last) {
      if (next.request_id !== last.request_id || next.model_digest !== last.model_digest
          || next.prompt_tokens !== last.prompt_tokens || next.max_output_tokens !== last.max_output_tokens
          || BigInt(next.revision) < BigInt(last.revision)
          || next.completion_tokens < last.completion_tokens) throw new Error('identity/counter regression');
      if (next.revision === last.revision) {
        if (JSON.stringify(next) !== JSON.stringify(last)) throw new Error('changed same-revision snapshot');
        return false;
      }
      if (last.state >= 4 || next.state < last.state) throw new Error('terminal/state regression');
      if ((next.state <= 3 && next.reserved_bytes !== last.reserved_bytes)
          || (last.state === 3 && (next.state !== 6 || next.reason !== last.reason))) {
        throw new Error('reservation/cancellation regression');
      }
    }
    this.last = next;
    return true;
  }
}
