/** Transport-independent contracts. Native download implementations must satisfy these checks. */
export interface RangeReply {
  status: number;
  contentRange: string | null;
  etag: string | null;
  contentLength: number | null;
}
export type ResumeDecision = 'append' | 'restart';

export function validateResumeResponse(reply: RangeReply, offset: number, expectedBytes: number, originalEtag: string | null): ResumeDecision {
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(expectedBytes) || offset < 0 || offset >= expectedBytes) {
    throw new Error('Invalid download offset.');
  }
  if (reply.status === 200) {
    if (reply.contentLength !== null && reply.contentLength !== expectedBytes) throw new Error('Unexpected model size.');
    return 'restart'; // A server that ignores Range must never be appended to a partial file.
  }
  if (reply.status !== 206) throw new Error(`Download response ${reply.status} is not a valid file range.`);
  const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(reply.contentRange ?? '');
  if (!match) throw new Error('Missing or invalid Content-Range.');
  const [start, end, total] = match.slice(1).map(Number);
  if (![start, end, total].every(Number.isSafeInteger) || start !== offset || total !== expectedBytes || end < start || end >= total) {
    throw new Error('Download range does not match the pinned model.');
  }
  if (originalEtag && reply.etag !== originalEtag) throw new Error('The download identity changed. Restart from the pinned source.');
  if (reply.contentLength !== null && reply.contentLength !== end - start + 1) throw new Error('Download range length mismatch.');
  return 'append';
}

export function estimateSeconds(bytes: number, megabitsPerSecond: number): number | null {
  if (!Number.isFinite(bytes) || bytes < 0 || !Number.isFinite(megabitsPerSecond) || megabitsPerSecond <= 0) return null;
  return Math.ceil(bytes * 8 / (megabitsPerSecond * 1_000_000));
}

export function installationSpaceBytes(incomingBytes: number, existingBytes = 0, scratchBytes = 256 * 1024 * 1024): number {
  if (![incomingBytes, existingBytes, scratchBytes].every(n => Number.isSafeInteger(n) && n >= 0)) throw new Error('Invalid disk-space estimate.');
  // Total space occupied during atomic replacement on the same filesystem.
  // A platform copy rather than rename must add another incomingBytes before starting.
  return incomingBytes + existingBytes + scratchBytes;
}

export const DOWNLOAD_STAGES = [
  { id: 'app', name: 'App, harness, judge and knowledge', enables: 'Browse the local library while the model downloads.' },
  { id: 'lite', name: 'Atlantic Lite', enables: 'Local chat after the complete model verifies and the runtime passes its self-test.' },
  { id: 'atlantic', name: 'Atlantic, optional', enables: 'Select the full model on a compatible device. Continue using Lite while it downloads.' },
  { id: 'vision', name: 'Image support, optional', enables: 'Add the matching vision projector only when image support is needed and validated.' },
] as const;
