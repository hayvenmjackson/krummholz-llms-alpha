/** Pinned public metadata, checked 2026-11-17. No artifact is approved for app release. */
export interface ModelPack {
  id: string;
  name: string;
  version: string;
  sizeBytes: number;
  downloadLabel: string;
  description: string;
  compatibility: string;
  repository: string;
  revision: string;
  filename: string;
  sha256: string;
  license: string;
  releaseStatus: 'integration-required';
  recommendedFor: 'phones-and-computers' | 'high-memory-computers';
}

export const MODEL_PACKS: readonly ModelPack[] = [
  {
    id: 'atlantic-lite-v8.1-q4', name: 'Atlantic Lite', version: '8.1',
    sizeBytes: 2741241728, downloadLabel: '2.74 GB',
    description: 'The smaller companion for everyday questions, writing and your knowledge library.',
    compatibility: 'Intended for phones and computers. Available memory and runtime checks required.',
    repository: 'krummholz-ai/atlantic-lite-4b-v8.1-GGUF',
    revision: '2bd4f6a56740ce8c1afe7ee61d3b41001433410e',
    filename: 'krummholz-atlantic-lite-4b-v8.1-Q4_K_M.gguf',
    sha256: '5dcb83f693151a4bf836541c2a54f9eeb0cbd49c769400c36debb9d1b3bad0cb',
    license: 'Apache-2.0', releaseStatus: 'integration-required',
    recommendedFor: 'phones-and-computers',
  },
  {
    id: 'atlantic-v8-q4', name: 'Atlantic', version: '8',
    sizeBytes: 16742036672, downloadLabel: '16.74 GB',
    description: 'The full model for computers with room for a larger local workspace.',
    compatibility: 'High-memory devices only. Model-card target: a GPU with 24 GB memory. App validation pending.',
    repository: 'krummholz-ai/atlantic-27b-v8-GGUF',
    revision: '12f9aef52e11595a055f51ad553f9a077ffaea45',
    filename: 'krummholz-atlantic-27b-v8-Q4_K_M.gguf',
    sha256: '56545e13dbc13182174185d1c7cea0cfe6fcfdd4c35d44f8d625a6cd4fa8befe',
    license: 'Apache-2.0', releaseStatus: 'integration-required',
    recommendedFor: 'high-memory-computers',
  },
];

export const KNOWLEDGE_PACK = Object.freeze({
  name: 'krummholz-gates-library-v0', asOf: '2026-11-09',
  sizeBytes: 4975273, passages: 2359, titles: 141,
  sha256: 'ed60b93d3f527b6a4f2ce6efed2e4f5d8b1fe0eddf48aede5d2c83e0e8891f29',
  bodySha256: '8125c49fe90588c5e1042d3ae5bce369ae677b5be0222ec7ceb62850d55af6c8',
  license: 'CC-BY-SA-4.0',
});

export function modelDownloadUrl(pack: ModelPack): string {
  if (!/^[a-f0-9]{40}$/.test(pack.revision) || !/^[a-f0-9]{64}$/.test(pack.sha256)) {
    throw new Error('Model release must have an immutable revision and SHA-256.');
  }
  return `https://huggingface.co/${pack.repository}/resolve/${pack.revision}/${encodeURIComponent(pack.filename)}`;
}
