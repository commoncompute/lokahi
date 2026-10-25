import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {decodeSessionDiagnostic, SessionObservation} from './session.ts';
const rows = JSON.parse(readFileSync(process.argv[2], 'utf8')) as unknown[];
const observation = new SessionObservation();
for (const row of rows) assert.equal(observation.accept(row), true);
assert.equal(observation.accept(rows.at(-1)), false);
const final = decodeSessionDiagnostic(rows.at(-1));
assert.equal(final.native_runtime_qualified, false);
assert.equal(final.completion_tokens, 8);
assert.equal(final.reserved_bytes, '0');
for (const changes of [{native_runtime_qualified: true}, {reserved_bytes: Number.MAX_SAFE_INTEGER},
  {revision: '18446744073709551616'}, {completion_tokens: -1}, {plaintext: 'forbidden'},
  {request_id: 'x\u0000y'}, {reserved_bytes: '1'}, {request_id: 'fixture\n'},
  {model_digest: final.model_digest + '\n'}, {revision: final.revision + '\n'}]) {
  assert.throws(() => decodeSessionDiagnostic({...final, ...changes}));
}
assert.throws(() => observation.accept(rows[0]));
const running = decodeSessionDiagnostic(rows[1]);
const reserveObservation = new SessionObservation();
reserveObservation.accept(running);
assert.throws(() => reserveObservation.accept({...running, revision: '100', reserved_bytes: '513'}));
const pending = decodeSessionDiagnostic(rows[3]);
const pendingObservation = new SessionObservation();
pendingObservation.accept(pending);
assert.throws(() => pendingObservation.accept({...final, revision: '100', state: 4, reason: 1}));
assert.throws(() => pendingObservation.accept({...final, revision: '100', reason: 5}));
console.log('TypeScript/native diagnostic parity passed; no model execution or serving transport');
