import { readFile, mkdir, copyFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const app = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const source = resolve(app, '../../data/library/krummholz-gates-library-v0.json');
const expected = 'ed60b93d3f527b6a4f2ce6efed2e4f5d8b1fe0eddf48aede5d2c83e0e8891f29';
const bytes = await readFile(source);
if (createHash('sha256').update(bytes).digest('hex') !== expected) {
  throw new Error('Knowledge corpus changed. Review provenance and update the catalog deliberately.');
}
await mkdir(resolve(app, 'public'), { recursive: true });
await copyFile(source, resolve(app, 'public/knowledge-store.json'));
console.log(`Prepared verified knowledge pack: ${bytes.length.toLocaleString('en-US')} bytes`);
