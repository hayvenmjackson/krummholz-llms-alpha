import { KNOWLEDGE_PACK } from './catalog.ts';

export interface KnowledgeDocument {
  id: string;
  title: string;
  section: string;
  text: string;
  url: string;
  license: string;
  revision: string | number | null;
}
export interface KnowledgeHit extends KnowledgeDocument { score: number }
interface Corpus {
  format: 'krummholz-library-v0';
  name: string;
  as_of: string;
  documents: KnowledgeDocument[];
  sha256?: string;
}

const STOP = new Set('a an and are as at be by for from has have in is it its of on or that the this to was were what when where which who why with how did does do'.split(' '));
const tokenize = (text: string): string[] => (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter(t => !STOP.has(t));

/** BM25 port of krummholz_llm/library.py. Search is retrieval, never generated truth. */
export class KnowledgeLibrary {
  readonly asOf: string;
  readonly name: string;
  readonly count: number;
  readonly documents: readonly KnowledgeDocument[];
  private readonly frequencies: Map<string, number>[];
  private readonly documentFrequency: Map<string, number>;
  private readonly lengths: number[];
  private readonly average: number;

  private constructor(data: Corpus) {
    this.asOf = data.as_of;
    this.name = data.name;
    this.documents = data.documents.map(d => Object.freeze({ ...d }));
    this.count = this.documents.length;
    this.documentFrequency = new Map();
    this.lengths = [];
    this.frequencies = this.documents.map(d => {
      const heading = tokenize(`${d.title} ${d.section}`);
      const words = [...heading, ...heading, ...tokenize(d.text)];
      const frequencies = new Map<string, number>();
      for (const word of words) frequencies.set(word, (frequencies.get(word) ?? 0) + 1);
      for (const word of frequencies.keys()) this.documentFrequency.set(word, (this.documentFrequency.get(word) ?? 0) + 1);
      this.lengths.push(words.length);
      return frequencies;
    });
    this.average = this.lengths.reduce((a, b) => a + b, 0) / Math.max(1, this.count);
  }

  /** Shape validation only. Use fromVerifiedBytes for downloaded or packaged data. */
  static fromData(input: unknown): KnowledgeLibrary {
    if (!input || typeof input !== 'object') throw new Error('Knowledge pack is missing.');
    const data = input as Partial<Corpus>;
    if (data.format !== 'krummholz-library-v0' || typeof data.name !== 'string' || typeof data.as_of !== 'string' || !Array.isArray(data.documents)) {
      throw new Error('Unknown knowledge pack format.');
    }
    const seen = new Set<string>();
    for (const doc of data.documents) {
      if (!doc || ['id', 'title', 'section', 'text', 'url', 'license'].some(key => typeof doc[key as keyof KnowledgeDocument] !== 'string') || !doc.id || !doc.text || seen.has(doc.id)) {
        throw new Error('Invalid or duplicate knowledge passage.');
      }
      if (doc.url && !/^https?:\/\//i.test(doc.url)) throw new Error('Knowledge source must be an HTTP(S) URL.');
      seen.add(doc.id);
    }
    return new KnowledgeLibrary(data as Corpus);
  }

  static async fromVerifiedBytes(bytes: Uint8Array, expectedSha256: string): Promise<KnowledgeLibrary> {
    if (!/^[a-f0-9]{64}$/.test(expectedSha256)) throw new Error('Invalid knowledge checksum.');
    if (!globalThis.crypto?.subtle) throw new Error('Integrity verification is unavailable. Open the preview in a current browser or the native app.');
    const digest = await crypto.subtle.digest('SHA-256', new Uint8Array(bytes));
    const hex = [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
    if (hex !== expectedSha256) throw new Error('Knowledge pack failed its integrity check.');
    return KnowledgeLibrary.fromData(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)));
  }

  search(query: string, k = 5, minScore = 1): KnowledgeHit[] {
    const terms = new Set(tokenize(query.slice(0, 4096)));
    if (!terms.size || !this.count) return [];
    const limit = Math.min(50, Math.max(0, Number.isFinite(k) ? Math.floor(k) : 5));
    const scored: { index: number; score: number }[] = [];
    this.frequencies.forEach((counts, index) => {
      let score = 0;
      for (const term of terms) {
        const frequency = counts.get(term) ?? 0;
        if (frequency) {
          const df = this.documentFrequency.get(term) ?? 0;
          const idf = Math.log(1 + (this.count - df + 0.5) / (df + 0.5));
          score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * this.lengths[index] / this.average));
        }
      }
      if (score >= minScore) scored.push({ index, score });
    });
    scored.sort((a, b) => b.score - a.score || (this.documents[a.index].id < this.documents[b.index].id ? -1 : 1));
    return scored.slice(0, limit).map(({ index, score }) => ({ ...this.documents[index], score: Math.round(score * 1000) / 1000 }));
  }

  evidence(id: string): (KnowledgeDocument & { asOf: string; semanticSupport: 'not-determined'; untrustedContent: true }) | undefined {
    const doc = this.documents.find(d => d.id === id);
    return doc ? { ...doc, asOf: this.asOf, semanticSupport: 'not-determined', untrustedContent: true } : undefined;
  }
}

export async function loadBundledLibrary(): Promise<KnowledgeLibrary> {
  const embedded = document.getElementById('atlantic-knowledge-data');
  let bytes: Uint8Array;
  if (embedded) {
    const decoded = atob(embedded.textContent?.trim() ?? '');
    bytes = Uint8Array.from(decoded, character => character.charCodeAt(0));
  } else {
    const response = await fetch('./knowledge-store.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error('The knowledge pack is unavailable. Repair the installation to continue.');
    bytes = new Uint8Array(await response.arrayBuffer());
  }
  return KnowledgeLibrary.fromVerifiedBytes(bytes, KNOWLEDGE_PACK.sha256);
}
