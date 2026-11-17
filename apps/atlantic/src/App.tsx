import { useEffect, useRef, useState, type FormEvent } from 'react';
import {
  ArrowDownToLine, ArrowRight, BookOpen, Check, CheckCheck,
  ChevronDown, ChevronRight, CircleHelp, Download, ExternalLink, FileText,
  HardDrive, Leaf, MessageSquare, Plus, Search,
  Settings, ShieldCheck, SlidersHorizontal, Sprout, Trash2, Undo2, WifiOff, X,
} from 'lucide-react';
import { KnowledgeLibrary, loadBundledLibrary } from './knowledge';
import { MODEL_PACKS } from './catalog';
import './App.css';

type Screen = 'chat' | 'knowledge' | 'learned' | 'models' | 'settings';
type Memory = { id: string; text: string; created: string; status: 'review' | 'kept' };
type Preferences = { provider: 'offline' | 'brave' | 'duckduckgo' };
type KnowledgeResult = {
  id: string; title: string; section: string; text: string;
  url: string; license: string; revision: unknown; score: number;
};
type PreviewPack = {
  id: string; name: string; version: string; sizeBytes: number;
  downloadLabel: string; description: string; compatibility: string;
};

const STORAGE_KEY = 'atlantic-design-preview-v1';
const DEFAULT_PREFS: Preferences = { provider: 'offline' };
const NAV = [
  { id: 'chat', label: 'Chat', Icon: MessageSquare },
  { id: 'knowledge', label: 'Knowledge', Icon: BookOpen },
  { id: 'learned', label: 'Learned today', Icon: Sprout },
  { id: 'models', label: 'Models', Icon: HardDrive },
  { id: 'settings', label: 'Settings', Icon: Settings },
] as const;

function initialState(): { memories: Memory[]; preferences: Preferences } {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    return {
      memories: Array.isArray(data.memories) ? data.memories.filter((m: Memory) =>
        typeof m.id === 'string' && typeof m.text === 'string' && m.text.length <= 2000 &&
        typeof m.created === 'string' && ['review', 'kept'].includes(m.status)) : [],
      preferences: ['offline', 'brave', 'duckduckgo'].includes(data.preferences?.provider)
        ? data.preferences : DEFAULT_PREFS,
    };
  } catch { return { memories: [], preferences: DEFAULT_PREFS }; }
}

function sourceUrl(url: string) {
  try { const parsed = new URL(url); return ['http:', 'https:'].includes(parsed.protocol) ? parsed.href : null; }
  catch { return null; }
}

function displayDate(iso: string) {
  const date = new Date(iso);
  return Number.isNaN(date.valueOf()) ? 'Saved in this preview' : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export default function App() {
  const [screen, setScreen] = useState<Screen>('chat');
  const [state] = useState(initialState);
  const [memories, setMemories] = useState<Memory[]>(state.memories);
  const [preferences, setPreferences] = useState<Preferences>(state.preferences);
  const [library, setLibrary] = useState<KnowledgeLibrary | null>(null);
  const [libraryError, setLibraryError] = useState('');
  const [query, setQuery] = useState('');
  const [expandedPassage, setExpandedPassage] = useState<string | null>(null);
  const [memoryTab, setMemoryTab] = useState<'review' | 'kept' | 'weights'>('review');
  const [newMemory, setNewMemory] = useState('');
  const [showMemoryForm, setShowMemoryForm] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const [undoMemory, setUndoMemory] = useState<Memory | null>(null);
  const [chatText, setChatText] = useState('');
  const [chatMessages, setChatMessages] = useState<string[]>([]);
  const [modal, setModal] = useState<'download' | 'reset' | null>(null);
  const [selectedPack, setSelectedPack] = useState<PreviewPack | null>(null);
  const [storageError, setStorageError] = useState(false);
  const modalRef = useRef<HTMLDialogElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const packs = MODEL_PACKS as readonly PreviewPack[];
  const pending = memories.filter(m => m.status === 'review');
  const kept = memories.filter(m => m.status === 'kept');
  const results = library && query.trim() ? library.search(query, 12) as KnowledgeResult[] : [];

  useEffect(() => {
    let active = true;
    loadBundledLibrary().then(data => {
      if (active) setLibrary(data);
    }).catch(() => {
      if (active) setLibraryError('The knowledge store is unavailable in this preview. The packaged file needs to be restored.');
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ memories, preferences })); setStorageError(false); }
    catch { setStorageError(true); }
  }, [memories, preferences]);

  useEffect(() => {
    if (modal && modalRef.current && !modalRef.current.open) modalRef.current.showModal();
    if (!modal && modalRef.current?.open) modalRef.current.close();
  }, [modal]);

  useEffect(() => { contentRef.current?.scrollTo({ top: 0 }); }, [screen]);

  function go(next: Screen) { setScreen(next); }
  function browse(term = '') { setQuery(term); setExpandedPassage(null); go('knowledge'); }
  function proposeMemory(event: FormEvent) {
    event.preventDefault();
    const text = newMemory.trim();
    if (!text) return;
    setMemories(items => [...items, { id: crypto.randomUUID(), text, created: new Date().toISOString(), status: 'review' }]);
    setNewMemory(''); setShowMemoryForm(false); setMemoryTab('review');
  }
  function forget(memory: Memory) { setUndoMemory(memory); setMemories(items => items.filter(m => m.id !== memory.id)); }
  function restore() {
    if (undoMemory) setMemories(items => [...items.filter(m => m.id !== undoMemory.id), undoMemory]);
    setUndoMemory(null);
  }
  function sendPreview(event: FormEvent) {
    event.preventDefault();
    if (!chatText.trim()) return;
    setChatMessages(items => [...items, chatText.trim()]); setChatText('');
  }
  function downloadPlan(pack: PreviewPack) { setSelectedPack(pack); setModal('download'); }

  return <div className="atlantic-app">
    <a className="skip-link" href="#main-content">Skip to content</a>
    <aside className="sidebar" aria-label="Main navigation">
      <a className="brand" href="#" onClick={e => { e.preventDefault(); go('chat'); }} aria-label="Atlantic home">
        <img src="./krummholz-mark.png" alt="" />
        <span><strong>Atlantic</strong><small>by krummholz<em>AI</em></small></span>
      </a>
      <button className="new-chat" onClick={() => { setChatMessages([]); go('chat'); }}><Plus size={18} /> New conversation</button>
      <div className="nav-label">Your workspace</div>
      <nav className="main-nav">
        {NAV.map(({ id, label, Icon }) => <button key={id} className={screen === id ? 'nav-item active' : 'nav-item'}
          aria-current={screen === id ? 'page' : undefined} onClick={() => go(id)}>
          <Icon size={19} /><span>{label}</span>{id === 'learned' && pending.length > 0 && <span className="nav-count">{pending.length}</span>}
        </button>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="local-note"><span className="local-note-icon"><WifiOff size={17}/></span><div><strong>A little more independent.</strong><p>Built around your own device.</p></div></div>
        <button className="sidebar-model" onClick={() => go('models')}><Download size={17}/><span>Set up your model</span><ChevronRight size={16}/></button>
        <div className="sidebar-footer"><span>Krummholz AI</span><span>Design preview</span></div>
      </div>
    </aside>

    <div className="main-shell">
      <header className="topbar">
        <div className="mobile-brand"><img src="./krummholz-mark.png" alt=""/><strong>Atlantic</strong></div>
        <div className="breadcrumb"><span>My workspace</span><ChevronRight size={14}/><strong>{NAV.find(item => item.id === screen)?.label}</strong></div>
        <button className="model-picker" onClick={() => go('models')} aria-label="View Atlantic Lite model setup">
          <span className="model-dot"/><span>Atlantic Lite</span><span className="model-picker-note">Preview</span><ChevronDown size={14}/>
        </button>
      </header>
      <div className="preview-notice" role="status"><span className="preview-dot"/><span><strong>Design preview.</strong> No model connected. Try the knowledge store and memory controls.</span></div>
      {storageError && <div className="error-notice" role="alert">This browser cannot save preview memories. Changes will last only while this page is open.</div>}

      <main id="main-content" ref={contentRef} className={`content content-${screen}`}>
        {screen === 'chat' && <section className="chat-screen" aria-label="Conversation">
          {chatMessages.length === 0 ? <div className="chat-welcome">
            <div className="welcome-mark"><img src="./krummholz-mark.png" alt=""/></div>
            <p className="eyebrow">Your model. Your knowledge. Your space.</p>
            <h1>Room to <em>think.</em></h1>
            <p className="welcome-copy">A thoughtful conversation, with a little more<br className="desktop-break"/> of your world close at hand.</p>
            <div className="starter-grid">
              <button className="starter" onClick={() => browse('Cultural Revolution')}><BookOpen size={20}/><span><strong>Explore what it knows</strong><small>Look inside the offline knowledge store</small></span><ArrowRight size={18}/></button>
              <button className="starter" onClick={() => { go('learned'); setShowMemoryForm(true); }}><Sprout size={20}/><span><strong>Make it a little more yours</strong><small>Add a preference for you to review</small></span><ArrowRight size={18}/></button>
            </div>
            <button className="setup-link" onClick={() => go('models')}>First time here? <span>See the model setup</span><ArrowRight size={14}/></button>
          </div> : <div className="conversation" aria-live="polite">
            {chatMessages.map((message, index) => <div className="conversation-pair" key={`${index}-${message.slice(0,10)}`}>
              <div className="user-message"><span className="message-author">You</span><p>{message}</p></div>
              <div className="preview-reply"><div className="reply-avatar"><img src="./krummholz-mark.png" alt=""/></div><div><span className="message-author">Preview notice</span><p>Your message has stayed here. This design preview is not connected to a language model, so it cannot generate an answer yet.</p><button className="text-button" onClick={() => browse(message)}>Search the bundled knowledge store <ArrowRight size={14}/></button></div></div>
            </div>)}
          </div>}
          <div className="composer-area">
            <form className="composer" onSubmit={sendPreview}>
              <label className="sr-only" htmlFor="message">Message Atlantic</label>
              <textarea id="message" rows={2} maxLength={8000} value={chatText} placeholder="What's on your mind?" onChange={e => setChatText(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!e.nativeEvent.isComposing) sendPreview(e); } }} />
              <div className="composer-tools"><button type="button" className="composer-knowledge" onClick={() => browse()}><BookOpen size={16}/><span>Knowledge store</span></button><span className="offline-tag"><WifiOff size={14}/><span>Offline</span></span><button className="send-button" type="submit" disabled={!chatText.trim()} aria-label="Try sending a preview message"><ArrowRight size={20}/></button></div>
            </form>
            <p className="composer-footnote">A good answer can start with “I don't know.” <span>Always check what matters.</span></p>
          </div>
        </section>}

        {screen === 'knowledge' && <section className="page knowledge-page">
          <div className="page-heading"><div><p className="eyebrow">Close at hand</p><h1>Your knowledge store.</h1><p className="page-intro">A dated history and sovereignty reference pack, available offline.</p></div><span className="quiet-badge"><WifiOff size={14}/> On this device</span></div>
          <div className="library-summary"><BookOpen size={21}/><span><strong>{library ? library.count.toLocaleString() : 'Loading'} passages</strong><span className="summary-divider">·</span>History and sovereignty</span><span className="library-date">{library ? `As of ${library.asOf}` : 'Opening the bundled store'}</span></div>
          <form className="knowledge-search" onSubmit={e => e.preventDefault()} role="search"><Search size={20}/><label className="sr-only" htmlFor="knowledge-query">Search the knowledge store</label><input id="knowledge-query" type="search" placeholder="Search a topic, place, or question" value={query} maxLength={500} onChange={e => { setQuery(e.target.value); setExpandedPassage(null); }}/>{query && <button type="button" className="icon-button" aria-label="Clear knowledge search" onClick={() => setQuery('')}><X size={17}/></button>}</form>
          {libraryError && <p className="error-notice" role="alert">{libraryError}</p>}
          {!query.trim() ? <div className="knowledge-explore"><div className="section-label">A few places to begin</div><div className="topic-grid">{[['Cultural Revolution', 'Cultural Revolution'], ['Taiwan', 'Taiwan'], ['Censorship', 'Censorship'], ['Kosovo', 'Kosovo']].map(([label, term]) => <button key={label} onClick={() => setQuery(term)}><span>{label}</span><ArrowRight size={18}/></button>)}</div><div className="knowledge-explanation"><FileText size={23}/><div><h2>Sources you can open.</h2><p>This first pack covers 141 titles. Every passage keeps its original source, revision, and license. Search finds relevant text; it does not certify that the text answers your question.</p></div></div></div> : <div className="knowledge-results" aria-live="polite">
            <div className="results-meta"><span>{results.length ? `${results.length} matching passages` : library ? 'No matching passages' : 'Opening knowledge store...'}</span><span>Search stays on this device</span></div>
            {results.length === 0 && library && <div className="empty-state compact"><Search size={26}/><h2>Try another phrase.</h2><p>This is a small, dated collection. A shorter query or a different topic may help.</p><button className="secondary-button" onClick={() => setQuery('')}>Browse topics</button></div>}
            {results.map(result => <article className={`passage ${expandedPassage === result.id ? 'expanded' : ''}`} key={result.id}>
              <button className="passage-heading" onClick={() => setExpandedPassage(expandedPassage === result.id ? null : result.id)} aria-expanded={expandedPassage === result.id} aria-controls={`passage-${result.id}`}><div><span className="source-label">Reference passage</span><h2>{result.title}</h2>{result.section && <span className="passage-section">{result.section}</span>}</div><ChevronDown size={18}/></button>
              <div id={`passage-${result.id}`}><p className={expandedPassage === result.id ? 'passage-text full' : 'passage-text'}>{result.text}</p>{expandedPassage === result.id && <div className="source-detail"><span>Revision: {String(result.revision || 'Not supplied')}</span><span>Collection date: {library?.asOf}</span><span>This is source material, not an Atlantic-generated answer.</span></div>}</div>
              <div className="passage-footer"><span>{result.license}</span>{sourceUrl(result.url) && <a href={sourceUrl(result.url)!} target="_blank" rel="noopener noreferrer">Open original source<ExternalLink size={13}/></a>}</div>
            </article>)}
          </div>}
          <div className="data-footnote"><ShieldCheck size={16}/><p>Reference text is treated as evidence, never as instructions. Reading a passage does not add it to personal memory or train the model.</p></div>
        </section>}

        {screen === 'learned' && <section className="page learned-page">
          <div className="page-heading"><div><p className="eyebrow">A little more you</p><h1>What I learned today.</h1><p className="page-intro">You decide what stays. Review it, change it, or let it go.</p></div><button className="primary-button" onClick={() => setShowMemoryForm(!showMemoryForm)}><Plus size={17}/> Add a memory</button></div>
          <div className="memory-tabs" role="tablist" aria-label="Learning records" onKeyDown={event => {
            const tabs = ['review', 'kept', 'weights'] as const;
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
            event.preventDefault();
            const current = tabs.indexOf(memoryTab);
            const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
              : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
            setMemoryTab(tabs[next]);
            document.getElementById(`tab-${tabs[next]}`)?.focus();
          }}>{(['review', 'kept', 'weights'] as const).map(tab => <button key={tab} role="tab" id={`tab-${tab}`} tabIndex={memoryTab === tab ? 0 : -1} aria-selected={memoryTab === tab} aria-controls="memory-panel" className={memoryTab === tab ? 'selected' : ''} onClick={() => setMemoryTab(tab)}>{tab === 'review' ? 'To review' : tab === 'kept' ? 'Remembered' : 'Dynamic weights'}{tab !== 'weights' && <span>{tab === 'review' ? pending.length : kept.length}</span>}</button>)}</div>
          {showMemoryForm && <form className="memory-form" onSubmit={proposeMemory}><div className="form-heading"><h2>What would you like it to remember?</h2><button className="icon-button" type="button" onClick={() => setShowMemoryForm(false)} aria-label="Close new memory form"><X size={18}/></button></div><label className="sr-only" htmlFor="new-memory">New memory</label><textarea autoFocus id="new-memory" rows={3} value={newMemory} maxLength={2000} placeholder="For example: I prefer short, practical explanations." onChange={e => setNewMemory(e.target.value)}/><div className="form-footer"><p>Saved only in this preview, after your review.</p><button className="primary-button" disabled={!newMemory.trim()} type="submit">Add to review <ArrowRight size={16}/></button></div></form>}
          <div id="memory-panel" role="tabpanel" aria-labelledby={`tab-${memoryTab}`}>
            {memoryTab === 'weights' ? <div className="weights-panel"><span className="large-icon"><SlidersHorizontal size={28}/></span><p className="eyebrow">A separate kind of learning</p><h2>Memory now. Weight changes, carefully.</h2><p>Remembering a preference and changing a model are different actions. A future weight update will show the exact model, the checks it passed, and a way to return to the previous version.</p><div className="integration-status"><span className="status-dot pending"/><strong>Native training and rollback are not connected in this preview.</strong></div><p className="secondary-copy">Keeping a memory here does not change either model's weights.</p></div> : (memoryTab === 'review' ? pending : kept).length === 0 ? <div className="empty-state"><span className="empty-art"><Sprout size={39} strokeWidth={1.3}/></span><h2>{memoryTab === 'review' ? 'Nothing waiting for you.' : 'A fresh start.'}</h2><p>{memoryTab === 'review' ? 'When there is something worth remembering, it belongs here for you to review.' : 'Memories you choose to keep will appear here. You can forget them at any time.'}</p><button className="text-button" onClick={() => setShowMemoryForm(true)}>Try adding a preference <ArrowRight size={15}/></button></div> : <div className="memory-list">{(memoryTab === 'review' ? pending : kept).map(memory => <article className="memory-card" key={memory.id}>
              <div className="memory-card-meta"><span className="memory-kind"><Sprout size={14}/> Personal memory</span><span>{displayDate(memory.created)}</span></div>
              {editing === memory.id ? <form onSubmit={e => { e.preventDefault(); if (!editText.trim()) return; setMemories(items => items.map(m => m.id === memory.id ? { ...m, text: editText.trim() } : m)); setEditing(null); }}><label className="sr-only" htmlFor="edited-memory">Edit memory</label><textarea id="edited-memory" rows={3} maxLength={2000} value={editText} onChange={e => setEditText(e.target.value)} autoFocus/><div className="memory-actions"><button className="primary-button" disabled={!editText.trim()} type="submit">Save edit</button><button className="secondary-button" type="button" onClick={() => setEditing(null)}>Cancel</button></div></form> : <><p className="memory-content">{memory.text}</p><p className="memory-origin">Added by you in this preview.</p><div className="memory-actions">{memory.status === 'review' ? <><button className="primary-button" onClick={() => setMemories(items => items.map(m => m.id === memory.id ? { ...m, status: 'kept' } : m))}><Check size={16}/> Keep</button><button className="secondary-button" onClick={() => { setEditing(memory.id); setEditText(memory.text); }}>Edit</button><button className="text-button muted-action" onClick={() => forget(memory)}>Skip</button></> : <><span className="kept-status"><CheckCheck size={16}/> Remembered</span><button className="text-button" onClick={() => { setEditing(memory.id); setEditText(memory.text); }}>Edit</button><button className="text-button muted-action" onClick={() => forget(memory)}><Trash2 size={15}/> Forget</button></>}</div></>}
            </article>)}</div>}
          </div>
          <div className="data-footnote"><ShieldCheck size={16}/><p>These controls demonstrate personal memory. They do not create training data, change weights, or upload your information.</p></div>
        </section>}

        {screen === 'models' && <section className="page models-page">
          <div className="page-heading"><div><p className="eyebrow">One app. A choice of models.</p><h1>Find your fit.</h1><p className="page-intro">Start with Lite. Add Atlantic when your device has room.</p></div></div>
          <div className="setup-summary"><span className="large-icon"><HardDrive size={24}/></span><div><h2>Simple setup, from inside the app.</h2><p>The release app will check available memory and storage before recommending a download. No device measurements have been taken in this preview.</p></div></div>
          <div className="model-grid">{packs.map((pack, index) => <article className={`model-card ${index === 0 ? 'recommended' : ''}`} key={pack.id}><div className="model-card-top"><span className="model-monogram">{index === 0 ? <Leaf size={25} strokeWidth={1.6}/> : <BookOpen size={25} strokeWidth={1.6}/>}</span><span className={index === 0 ? 'model-tag recommended-tag' : 'model-tag'}>{index === 0 ? 'Start here' : 'For capable computers'}</span></div><h2>{pack.name}</h2><p className="model-version">{pack.version} · GGUF</p><p className="model-description">{pack.description}</p><div className="model-size"><strong>{(pack.sizeBytes / 1e9).toFixed(2)}<span>GB</span></strong><span>Model download</span></div><p className="model-compatibility">{pack.compatibility}</p><button className={index === 0 ? 'primary-button' : 'secondary-button'} onClick={() => downloadPlan(pack)}><ArrowDownToLine size={17}/> View download plan</button><p className="model-install-status">Not installed in this preview</p></article>)}</div>
          <div className="download-explanation"><div><Download size={21}/><h2>Download once. Pick up where you left off.</h2></div><p>The proposed setup keeps partial downloads, checks every file, and makes a complete Lite model available before any optional larger model. Staging reduces the wait to a useful first model; it cannot reduce the model's actual size.</p></div>
          <div className="platform-row" aria-label="Target platforms"><span>Designed for</span><strong>Windows</strong><strong>macOS</strong><strong>Linux</strong><strong>iOS</strong><strong>Android</strong></div>
        </section>}

        {screen === 'settings' && <section className="page settings-page">
          <div className="page-heading"><div><p className="eyebrow">Your call</p><h1>Make yourself at home.</h1><p className="page-intro">Clear choices about what stays here and what goes online.</p></div></div>
          <section className="settings-section"><div className="settings-heading"><WifiOff size={20}/><div><h2>Search preference</h2><p>Search is optional. The offline knowledge store is always separate.</p></div></div><fieldset className="provider-options"><legend className="sr-only">Preferred search provider for the app design</legend>{[{ id: 'offline', name: 'Stay offline', detail: 'Use the knowledge collection on this device.' }, { id: 'brave', name: 'Brave Search', detail: 'Integrated web search, when connected.' }, { id: 'duckduckgo', name: 'DuckDuckGo', detail: 'Open a search in your browser, when connected.' }].map(provider => <label className={`provider-option ${preferences.provider === provider.id ? 'chosen' : ''}`} key={provider.id}><input type="radio" name="provider" value={provider.id} checked={preferences.provider === provider.id} onChange={() => setPreferences({ provider: provider.id as Preferences['provider'] })}/><span><strong>{provider.name}</strong><small>{provider.detail}</small></span>{preferences.provider === provider.id && <Check size={18}/>}</label>)}</fieldset><p className="settings-note">Preference preview only. No search provider is connected and no query is sent. In the finished app, using web search will send your search query to the chosen service.</p></section>
          <section className="settings-section"><div className="settings-heading"><ShieldCheck size={20}/><div><h2>Data on this device</h2><p>A clear record, with an easy way to start again.</p></div></div><div className="settings-row"><div><strong>Preview memories</strong><p>{kept.length} remembered, {pending.length} waiting for review. Saved in this browser's local storage.</p></div><button className="secondary-button" onClick={() => go('learned')}>Review</button></div><div className="settings-row"><div><strong>Conversation in this preview</strong><p>Kept only while this page is open. No language model is running.</p></div></div><div className="settings-row"><div><strong>Clear preview data</strong><p>Remove memories and reset your search preference.</p></div><button className="secondary-button danger-button" onClick={() => setModal('reset')}>Clear data</button></div></section>
          <section className="settings-section"><div className="settings-heading"><CircleHelp size={20}/><div><h2>About this design</h2><p>Atlantic by Krummholz AI.</p></div></div><div className="about-detail"><p>The interface is a working design preview. Knowledge search and personal memory controls work here. Native model execution, the judge, gated answers, model downloads, and dynamic weights still require integration and device testing.</p><a className="text-button" href="https://github.com/hayvenmjackson/krummholz-llms-alpha" target="_blank" rel="noopener noreferrer">Project source <ExternalLink size={14}/></a></div></section>
        </section>}
      </main>
    </div>

    <nav className="mobile-nav" aria-label="Main navigation">{NAV.map(({ id, label, Icon }) => <button key={id} className={screen === id ? 'active' : ''} aria-current={screen === id ? 'page' : undefined} onClick={() => go(id)}><span className="mobile-nav-icon"><Icon size={21}/>{id === 'learned' && pending.length > 0 && <i/>}</span><span>{id === 'learned' ? 'Learned' : label}</span></button>)}</nav>
    {undoMemory && <div className="toast" role="status"><span>Memory removed.</span><button onClick={restore}><Undo2 size={15}/> Undo</button><button className="icon-button" aria-label="Dismiss undo notification" onClick={() => setUndoMemory(null)}><X size={16}/></button></div>}
    <dialog ref={modalRef} className="app-dialog" onCancel={() => setModal(null)} onClose={() => setModal(null)}>
      <div className="dialog-heading"><span className="eyebrow">{modal === 'reset' ? 'A fresh start' : 'The first-run experience'}</span><button className="icon-button" onClick={() => setModal(null)} aria-label="Close dialog"><X size={20}/></button></div>
      {modal === 'download' && selectedPack ? <><h2>{selectedPack.name}, step by step.</h2><p className="dialog-intro">This is the setup plan, not a live download.</p><ol className="download-steps"><li><span>1</span><div><strong>Check this device</strong><p>Measure available memory and disk space. Recommend a model that fits.</p></div></li><li><span>2</span><div><strong>Prepare the essentials</strong><p>Install the runtime, knowledge store, judge, and required gates.</p></div></li><li><span>3</span><div><strong>Download {selectedPack.name}</strong><p>{(selectedPack.sizeBytes / 1e9).toFixed(2)} GB for the model. Pause and resume with verification before installation.</p></div></li><li><span>4</span><div><strong>Check, then start a conversation</strong><p>Verify the complete files and run startup checks before enabling chat.</p></div></li></ol><div className="dialog-note"><WifiOff size={18}/><p>A complete model is needed before it can answer offline. Optional models and image support can download later.</p></div><button className="primary-button dialog-done" onClick={() => setModal(null)}>Got it <Check size={17}/></button></> : modal === 'reset' ? <><h2>Clear your preview data?</h2><p className="dialog-intro">This removes the memories you added to this preview and restores the offline search preference. The bundled knowledge collection stays available.</p><div className="dialog-actions"><button className="secondary-button" onClick={() => setModal(null)}>Keep my data</button><button className="primary-button" onClick={() => { setMemories([]); setPreferences(DEFAULT_PREFS); setUndoMemory(null); setChatMessages([]); setModal(null); }}>Clear preview data</button></div></> : null}
    </dialog>
  </div>;
}
