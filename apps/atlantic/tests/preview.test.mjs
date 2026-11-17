import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';
import { JSDOM, VirtualConsole } from 'jsdom';

// Exercises the actual standalone bundle. This is a DOM test, not a visual or device test.
test('standalone preview loads its real offline corpus and completes memory review', async () => {
  const html = await readFile(new URL('../atlantic-design-preview.html', import.meta.url), 'utf8');
  const errors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on('jsdomError', error => { if (error.type !== 'css-parsing') errors.push(error.message); });
  const dom = new JSDOM(html, {
    url: 'https://offline-preview.test/', runScripts: 'outside-only', pretendToBeVisual: true, virtualConsole,
    beforeParse(window) {
      Object.defineProperty(window, 'crypto', { value: webcrypto });
      window.TextDecoder = TextDecoder;
      window.TextEncoder = TextEncoder;
      window.fetch = () => { throw new Error('Standalone preview attempted a network request.'); };
      window.HTMLElement.prototype.scrollTo = () => {};
      window.HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
      window.HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); this.dispatchEvent(new window.Event('close')); };
      window.addEventListener('error', event => errors.push(event.message));
    },
  });
  const { window } = dom;
  const document = window.document;
  const settle = async (condition, label) => {
    const deadline = Date.now() + 5000;
    while (!condition() && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 15));
    assert.ok(condition(), label);
  };
  const click = async (selector, text) => {
    const button = [...document.querySelectorAll(selector)].find(node => node.textContent.includes(text));
    assert.ok(button, `Control exists: ${text}`);
    button.click();
    await new Promise(resolve => setTimeout(resolve, 20));
  };
  const setInput = (selector, value) => {
    const input = document.querySelector(selector);
    assert.ok(input, `Input exists: ${selector}`);
    const prototype = input.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, value);
    input.dispatchEvent(new window.Event('input', { bubbles: true }));
  };
  try {
    const module = document.querySelector('script[type="module"]');
    assert.ok(module && !module.src, 'The executable bundle is embedded.');
    assert.equal(document.querySelectorAll('link[rel="stylesheet"]').length, 0);
    window.eval(module.textContent);
    await settle(() => document.querySelector('.preview-notice'), 'The interface renders.');
    assert.match(document.querySelector('.preview-notice').textContent, /No model connected/);
    assert.ok([...document.images].every(image => image.src.startsWith('data:')), 'Every rendered logo is embedded.');

    await click('.main-nav button', 'Knowledge');
    await settle(() => document.querySelector('.library-summary')?.textContent.includes('2,359'), 'The real corpus loads and verifies without networking.');
    await click('.topic-grid button', 'Cultural Revolution');
    await settle(() => document.querySelectorAll('.passage').length > 0, 'The real knowledge store produces results.');
    await click('.passage-heading', 'Cultural Revolution');
    assert.ok(document.querySelector('.passage.expanded .source-detail'));
    assert.match(document.querySelector('.passage-footer').textContent, /CC BY-SA/);
    assert.match(document.querySelector('.passage-footer a').href, /^https:\/\//);

    await click('.main-nav button', 'Learned today');
    await click('button', 'Add a memory');
    setInput('#new-memory', 'Example preference: short practical explanations.');
    await settle(() => !document.querySelector('.memory-form button[type="submit"]').disabled, 'Entered memory is reviewable.');
    document.querySelector('.memory-form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
    await settle(() => document.querySelector('.memory-card'), 'Memory enters review.');
    await click('.memory-actions button', 'Edit');
    setInput('#edited-memory', '');
    await settle(() => document.querySelector('.memory-card button[type="submit"]').disabled, 'Empty edit cannot discard a candidate.');
    setInput('#edited-memory', 'Example preference: concise explanations with units.');
    await new Promise(resolve => setTimeout(resolve, 25));
    document.querySelector('.memory-card form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
    await settle(() => document.querySelector('.memory-content')?.textContent.includes('with units'), 'Edited memory is retained.');
    await click('.memory-actions button', 'Keep');
    await click('.memory-tabs button', 'Remembered');
    assert.match(document.querySelector('.memory-content').textContent, /with units/);
    await click('.memory-actions button', 'Forget');
    assert.equal(document.querySelectorAll('.memory-card').length, 0);
    await click('.toast button', 'Undo');
    assert.match(document.querySelector('.memory-content').textContent, /with units/);
    assert.equal(JSON.parse(window.localStorage.getItem('atlantic-design-preview-v1')).memories[0].status, 'kept');
    await click('.memory-tabs button', 'Dynamic weights');
    assert.match(document.querySelector('.weights-panel').textContent, /not connected/);

    await click('.main-nav button', 'Models');
    assert.match(document.querySelector('.model-grid').textContent, /2\.74/);
    assert.match(document.querySelector('.model-grid').textContent, /16\.74/);
    await click('.model-card button', 'View download plan');
    assert.ok(document.querySelector('dialog[open]'));
    assert.match(document.querySelector('dialog').textContent, /not a live download/);
    await click('dialog button', 'Got it');

    await click('.mobile-nav button', 'Chat');
    setInput('#message', 'Synthetic preview message');
    await new Promise(resolve => setTimeout(resolve, 25));
    document.querySelector('.composer').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
    await settle(() => document.querySelector('.preview-reply'), 'Chat tells the truth about unavailable inference.');
    assert.match(document.querySelector('.preview-reply').textContent, /cannot generate an answer/);
    assert.deepEqual(errors, []);
  } finally { dom.window.close(); }
});
