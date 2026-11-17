import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { KnowledgeLibrary } from '../src/knowledge.ts';
import { KNOWLEDGE_PACK, MODEL_PACKS, modelDownloadUrl } from '../src/catalog.ts';
import { validateResumeResponse, estimateSeconds, installationSpaceBytes } from '../src/download-plan.ts';

const corpusPath = new URL('../../../data/library/krummholz-gates-library-v0.json', import.meta.url);

test('real knowledge bytes pass the published raw-file checksum and preserve attribution', async () => {
  const library = await KnowledgeLibrary.fromVerifiedBytes(await readFile(corpusPath), KNOWLEDGE_PACK.sha256);
  assert.equal(library.count, 2359);
  assert.equal(library.asOf, '2026-11-09');
  const result = library.search('Cultural Revolution', 1)[0];
  assert.ok(result);
  const evidence = library.evidence(result.id)!;
  assert.ok(evidence.url.startsWith('https://'));
  assert.ok(evidence.license.includes('BY-SA'));
  assert.equal(evidence.semanticSupport, 'not-determined');
  assert.equal(evidence.untrustedContent, true);
});

test('BM25 port agrees with the real Python store for varied questions', async () => {
  const library = KnowledgeLibrary.fromData(JSON.parse(await readFile(corpusPath, 'utf8')));
  const queries = ['Cultural Revolution', 'Taiwan', 'Great Chinese Famine', 'Kosovo', 'human rights censorship', 'the and of', 'zzunlikelynomatch'];
  const python = 'import json,sys; from krummholz_llm.library import Library; lib=Library.load(sys.argv[1]); print(json.dumps([lib.search(q) for q in json.loads(sys.argv[2])]))';
  const actual = JSON.parse(execFileSync(process.platform === 'win32' ? 'python' : 'python3', ['-c', python, fileURLToPath(corpusPath), JSON.stringify(queries)], { cwd: new URL('../../../', import.meta.url), encoding: 'utf8' }));
  const port = queries.map(q => library.search(q).map(d => [d.id, d.score]));
  assert.deepEqual(port, actual);
});

test('corrupt knowledge cannot silently become an unverified library', async () => {
  const bytes = await readFile(corpusPath);
  bytes[100] ^= 1;
  await assert.rejects(KnowledgeLibrary.fromVerifiedBytes(bytes, KNOWLEDGE_PACK.sha256), /integrity/);
});

test('unknown packs, duplicate passage IDs and executable source URLs are rejected', () => {
  assert.throws(() => KnowledgeLibrary.fromData({ format: 'other' }), /format/);
  const doc = { id: 'one', title: 'Title', section: '', text: 'content', url: 'https://example.test/source', license: 'CC0', revision: null };
  const base = { format: 'krummholz-library-v0', name: 'test', as_of: '2026-11-17' };
  assert.throws(() => KnowledgeLibrary.fromData({ ...base, documents: [doc, doc] }), /duplicate/);
  assert.throws(() => KnowledgeLibrary.fromData({ ...base, documents: [{ ...doc, url: 'javascript:alert(1)' }] }), /HTTP/);
});

test('resumption validates offset, file identity and byte range', () => {
  const response = { status: 206, contentRange: 'bytes 50-99/100', etag: 'stable', contentLength: 50 };
  assert.equal(validateResumeResponse(response, 50, 100, 'stable'), 'append');
  assert.throws(() => validateResumeResponse({ ...response, contentRange: 'bytes 0-49/100' }, 50, 100, 'stable'), /range/);
  assert.throws(() => validateResumeResponse({ ...response, etag: 'changed' }, 50, 100, 'stable'), /identity/);
  assert.throws(() => validateResumeResponse({ ...response, contentRange: 'bytes 50-99/101' }, 50, 100, 'stable'), /range/);
  assert.throws(() => validateResumeResponse({ ...response, contentLength: 49 }, 50, 100, 'stable'), /length/);
});

test('a server ignoring Range triggers restart, never append', () => {
  assert.equal(validateResumeResponse({ status: 200, contentRange: null, etag: null, contentLength: 100 }, 50, 100, null), 'restart');
  assert.throws(() => validateResumeResponse({ status: 416, contentRange: null, etag: null, contentLength: 0 }, 50, 100, null), /valid file range/);
});

test('download calculations respect actual model bytes without promising throughput', () => {
  assert.equal(estimateSeconds(MODEL_PACKS[0].sizeBytes, 50), 439);
  assert.equal(estimateSeconds(MODEL_PACKS[1].sizeBytes, 10), 13394);
  assert.equal(estimateSeconds(10, 0), null);
  assert.equal(installationSpaceBytes(100, 80, 10), 190);
  assert.throws(() => installationSpaceBytes(-1), /Invalid/);
  for (const pack of MODEL_PACKS) {
    assert.match(modelDownloadUrl(pack), new RegExp(`/resolve/${pack.revision}/`));
    assert.equal(pack.releaseStatus, 'integration-required');
  }
});
