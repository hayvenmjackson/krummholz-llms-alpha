import { readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
let html = await readFile(resolve(root, 'dist/index.html'), 'utf8');
const logo = (await readFile(resolve(root, 'public/krummholz-mark.png'))).toString('base64');
const scripts = [...html.matchAll(/<script\b[^>]*\bsrc="([^"]+)"[^>]*><\/script>/g)];
for (const match of scripts) {
  let js = await readFile(resolve(root, 'dist', match[1]), 'utf8');
  js = js.replaceAll('"./krummholz-mark.png"', '__atlanticStandaloneBrand_20261117');
  js = `const __atlanticStandaloneBrand_20261117 = "data:image/png;base64,${logo}";\n${js}`;
  // A callback is essential: minified source may contain literal $&, $' or $`.
  // String replacement syntax would interpret those and corrupt the executable.
  html = html.replace(match[0], () => `<script type="module">${js.replaceAll('</script', '<\\/script')}</script>`);
}
const styles = [...html.matchAll(/<link\b[^>]*\brel="stylesheet"[^>]*\bhref="([^"]+)"[^>]*>/g)];
for (const match of styles) {
  const css = await readFile(resolve(root, 'dist', match[1]), 'utf8');
  html = html.replace(match[0], () => `<style>${css}</style>`);
}
html = html.replaceAll('./krummholz-mark.png', `data:image/png;base64,${logo}`);
html = html.replaceAll('/krummholz-mark.png', `data:image/png;base64,${logo}`);
const corpus = (await readFile(resolve(root, 'public/knowledge-store.json'))).toString('base64');
html = html.replace('</head>', `<script type="text/plain" id="atlantic-knowledge-data">${corpus}</script></head>`);
const notices = await readFile(resolve(root, 'docs/THIRD_PARTY_NOTICES.txt'), 'utf8');
html = html.replace('</head>', () => `<script type="text/plain" id="atlantic-third-party-notices">${notices.replaceAll('</script', '<\\/script')}</script></head>`);
await writeFile(resolve(root, 'atlantic-design-preview.html'), html);
console.log('Created standalone offline design preview with the original verified knowledge pack.');
