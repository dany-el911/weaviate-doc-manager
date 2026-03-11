import React, { useMemo, useState, useEffect, useRef, useCallback } from "react";
import type { ChangeEvent } from "react";
import {
  Upload,
  Database,
  FileText,
  Plus,
  AlertCircle,
  CheckCircle,
  Loader,
  Trash2,
  X,
  Search,
  FolderOpen,
  Layers,
} from "lucide-react";

// --- TYPES ---

interface WeaviateClass { class?: string; name?: string; description?: string; }
interface StatusState { type: "error" | "success" | "loading" | ""; message: string; }
interface ConfigState { weaviateHost: string; weaviatePort: string; ollamaUrl: string; embedModel: string; }
interface ProgressData { stage: string; current: number; total: number; percent: number; error?: string; currentFile?: string; }

// ─── STATUS TOAST ────────────────────────────────────────────────────────────

const StatusToast: React.FC<{ status: StatusState; onDismiss: () => void; autoHideMs?: number }> = ({ status, onDismiss, autoHideMs = 0 }) => {
  const hasMessage = Boolean(status.message);

  React.useEffect(() => {
    if (!hasMessage || !autoHideMs) return;
    const timer = setTimeout(() => onDismiss(), autoHideMs);
    return () => clearTimeout(timer);
  }, [hasMessage, autoHideMs, onDismiss]);

  if (!hasMessage) return null;

  const map = {
    error:   { border: "border-l-red-500",    icon: <AlertCircle className="w-4 h-4 text-red-400" />,                       text: "text-red-200"   },
    success: { border: "border-l-green-500",  icon: <CheckCircle className="w-4 h-4 text-green-400" />,                     text: "text-green-200" },
    loading: { border: "border-l-yellow-400", icon: <Loader className="w-4 h-4 text-yellow-400 animate-spin" />,            text: "text-yellow-200" },
    "":      { border: "",                    icon: null,                                                                    text: ""               },
  };
  const s = map[status.type];

  return (
    <div className={`fixed bottom-6 right-6 z-50 flex items-start gap-3 bg-[#1c1c1c] border border-zinc-700 border-l-4 ${s.border} rounded-xl px-4 py-3 shadow-2xl max-w-sm`}>
      <div className="shrink-0 mt-0.5">{s.icon}</div>
      <p className={`text-sm flex-1 leading-snug ${s.text}`}>{status.message}</p>
      <button onClick={onDismiss} className="shrink-0 text-zinc-600 hover:text-zinc-300 transition ml-1">
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
};

// ─── MODAL ───────────────────────────────────────────────────────────────────

const Modal: React.FC<{
  open: boolean; title: string; description?: string;
  onClose: () => void; children: React.ReactNode; width?: string;
}> = ({ open, title, description, onClose, children, width = "max-w-lg" }) => {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center px-4">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-md" onClick={onClose} />
      <div className={`relative w-full ${width} bg-[#111] border border-zinc-800 rounded-2xl shadow-2xl overflow-hidden`}>
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-yellow-400/60 to-transparent" />
        <div className="px-6 pt-6 pb-0 flex items-start justify-between">
          <div>
            <h2 className="text-base font-semibold text-zinc-100">{title}</h2>
            {description && <p className="text-xs text-zinc-500 mt-1">{description}</p>}
          </div>
          <button onClick={onClose} className="w-7 h-7 flex items-center justify-center rounded-lg text-zinc-600 hover:text-zinc-200 hover:bg-zinc-800 transition">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="p-6">{children}</div>
      </div>
    </div>
  );
};

// ─── BUTTONS ─────────────────────────────────────────────────────────────────

const Btn: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "danger" }> = ({
  variant = "primary", className = "", children, ...props
}) => {
  const base = "inline-flex items-center justify-center gap-1.5 text-sm font-medium rounded-lg transition active:scale-[0.97] disabled:opacity-40 disabled:cursor-not-allowed";
  const variants = {
    primary: "bg-yellow-400 text-black px-3.5 py-2 hover:bg-yellow-300",
    ghost:   "border border-zinc-700 text-zinc-300 px-3.5 py-2 hover:border-zinc-500 hover:text-white hover:bg-zinc-800/50",
    danger:  "border border-red-800/60 text-red-400 px-3 py-1.5 hover:bg-red-950/40 hover:border-red-600/60",
  };
  return <button {...props} className={`${base} ${variants[variant]} ${className}`}>{children}</button>;
};

// ─── UPLOAD PROGRESS ─────────────────────────────────────────────────────────

const UploadLoader: React.FC<{
  uploadId: string; onCancel: () => void; onComplete: () => void; onError: (m: string) => void;
}> = ({ uploadId, onCancel, onComplete, onError }) => {
  const [p, setP] = useState<ProgressData>({ stage: "starting", current: 0, total: 0, percent: 0 });

  useEffect(() => {
    let alive = true;
    const iv = setInterval(async () => {
      try {
        const res = await fetch(`http://127.0.0.1:5001/api/upload-progress/${uploadId}`);
        if (!res.ok) return;
        const d: ProgressData = await res.json();
        if (alive) setP(d);
        if (d.stage === "done")      { clearInterval(iv); setTimeout(() => alive && onComplete(), 500); }
        if (d.stage === "cancelled") { clearInterval(iv); setTimeout(() => alive && onComplete(), 300); }
        if (d.stage === "error")     { clearInterval(iv); if (alive) onError(d.error || "Errore sconosciuto"); }
      } catch { /* noop */ }
    }, 600);
    return () => { alive = false; clearInterval(iv); };
  }, [uploadId, onComplete, onError]);

  const labels: Record<string, string> = {
    starting: "Avvio...", processing: "Indicizzazione...", done: "Completato", cancelled: "Annullato", error: "Errore",
  };

  return (
    <div className="fixed inset-0 z-[70] flex items-end justify-end p-6 pointer-events-none">
      <div className="pointer-events-auto w-80 bg-[#111] border border-zinc-800 rounded-2xl shadow-2xl overflow-hidden">
        <div className="h-1 bg-zinc-900 relative">
          <div className="absolute inset-y-0 left-0 bg-yellow-400 transition-all duration-500 rounded-full" style={{ width: `${p.percent}%` }} />
        </div>
        <div className="p-5">
          <div className="flex items-center gap-3 mb-3">
            <Loader className="w-4 h-4 animate-spin text-yellow-400 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-zinc-100">{labels[p.stage] ?? "In corso..."}</p>
              <p className="text-xs text-zinc-500 truncate">{p.currentFile || `File ${p.current} di ${p.total}`}</p>
            </div>
            <span className="text-sm font-bold text-yellow-400 tabular-nums">{p.percent}%</span>
          </div>
          <button
            onClick={onCancel}
            className="w-full text-xs text-red-400 hover:text-red-300 border border-red-900/50 hover:border-red-700/50 rounded-lg py-1.5 transition"
          >
            Annulla
          </button>
        </div>
      </div>
    </div>
  );
};

// ─── MAIN ────────────────────────────────────────────────────────────────────

const WeaviateDocumentManager: React.FC = () => {
  const [collections, setCollections] = useState<WeaviateClass[]>([]);
  const [collectionQuery, setCollectionQuery] = useState("");
  const [selectedCollection, setSelectedCollection] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [newCollectionDescription, setNewCollectionDescription] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusState>({ type: "", message: "" });
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);

  const uploadAbortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const config: ConfigState = {
    weaviateHost: "127.0.0.1", weaviatePort: "8080",
    ollamaUrl: "http://127.0.0.1:11434", embedModel: "qwen3-embedding:4b",
  };

  const loadCollections = useCallback(async () => {
    try {
      const res = await fetch(`http://${config.weaviateHost}:${config.weaviatePort}/v1/schema`);
      if (!res.ok) throw new Error();
      const data = await res.json() as { classes?: WeaviateClass[] };
      setCollections((data.classes ?? []).filter(c => {
        const n = c.class || c.name || "";
        return !n.startsWith("ELYSIA_") && !n.startsWith("_");
      }));
    } catch {
      setStatus({ type: "error", message: "Impossibile connettersi a Weaviate" });
    }
  }, [config.weaviateHost, config.weaviatePort]);

  useEffect(() => { void loadCollections(); }, [loadCollections]);

  const normalized = useMemo(() =>
    collections.map(c => ({
      name: (c.class || c.name || "").trim(),
      description: (c.description || "").trim(),
    }))
      .filter(c => c.name.length > 0).sort((a, b) => a.name.localeCompare(b.name)),
    [collections]
  );

  const visible = useMemo(() => {
    const q = collectionQuery.trim().toLowerCase();
    return q ? normalized.filter(c => c.name.toLowerCase().includes(q)) : normalized;
  }, [normalized, collectionQuery]);

  const fmt = (bytes: number) => {
    const kb = bytes / 1024;
    if (kb < 1024) return `${Math.round(kb)} KB`;
    const mb = kb / 1024;
    return mb < 1024 ? `${mb.toFixed(1)} MB` : `${(mb / 1024).toFixed(1)} GB`;
  };

  const openUpload = (pre?: string) => {
    setStatus(s => s.type === "success" ? { type: "", message: "" } : s);
    if (pre) setSelectedCollection(pre);
    setUploadOpen(true);
    setTimeout(() => fileInputRef.current?.focus(), 50);
  };

  const handleFiles = (e: ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const sel = Array.from(e.target.files);
    setFiles(prev => {
      const keys = new Set(prev.map(f => `${f.name}::${f.size}`));
      return [...prev, ...sel.filter(f => !keys.has(`${f.name}::${f.size}`))];
    });
    e.target.value = "";
  };

  const openCreate = () => {
    setNewCollectionName("");
    setNewCollectionDescription("");
    setCreateOpen(true);
  };

  const closeCreate = () => {
    setCreateOpen(false);
    setNewCollectionName("");
    setNewCollectionDescription("");
  };

  const createCollection = async () => {
    const name = newCollectionName.trim().toUpperCase();
    const description = newCollectionDescription.trim();
    if (!name) { setStatus({ type: "error", message: "Inserisci un nome" }); return; }
    setStatus({ type: "loading", message: "Creazione in corso..." });
    try {
      const res = await fetch("http://localhost:5001/api/create-collection", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description, config }),
      });
      if (!res.ok) throw new Error("Errore creazione");
      setStatus({ type: "success", message: `Collezione "${name}" creata` });
      setNewCollectionName(""); setNewCollectionDescription(""); setSelectedCollection(name); setCreateOpen(false);
      void loadCollections();
    } catch (e: unknown) {
      setStatus({ type: "error", message: e instanceof Error ? e.message : "Errore" });
    }
  };

  const deleteCollection = async (name: string) => {
    if (!window.confirm(`Eliminare "${name}"?`)) return;
    try {
      const res = await fetch("http://127.0.0.1:5001/api/delete-collection", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) throw new Error();
      if (selectedCollection === name) setSelectedCollection("");
      setStatus({ type: "success", message: `"${name}" eliminata` });
      void loadCollections();
    } catch {
      setStatus({ type: "error", message: "Errore durante l'eliminazione" });
    }
  };

  const startUpload = async () => {
    if (!selectedCollection || files.length === 0) return;
    const id = crypto.randomUUID?.() ?? `up_${Date.now()}`;
    setUploadId(id);
    const ctrl = new AbortController();
    uploadAbortRef.current = ctrl;
    const fd = new FormData();
    fd.append("collection", selectedCollection);
    fd.append("config", JSON.stringify(config));
    fd.append("uploadId", id);
    files.forEach(f => fd.append("files", f));
    setUploadOpen(false);
    fetch("http://127.0.0.1:5001/api/upload-documents", { method: "POST", body: fd, signal: ctrl.signal })
      .then(async r => { const d = await r.json(); if (!r.ok) throw new Error(d.error || "Errore upload"); })
      .catch(e => { if (e.name === "AbortError") return; setUploadId(null); setStatus({ type: "error", message: e.message }); });
  };

  const cancelUpload = async () => {
    uploadAbortRef.current?.abort();
    const id = uploadId;
    setUploadId(null);
    if (!id) return;
    try {
      const res = await fetch("http://127.0.0.1:5001/api/cancel-upload", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uploadId: id }),
      });
      const d = await res.json();
      setStatus({ type: "success", message: `Annullato. Rimossi ${d.deleted ?? 0} file.` });
    } catch { setStatus({ type: "", message: "" }); }
  };

  return (
    <>
      {uploadId && (
        <UploadLoader uploadId={uploadId} onCancel={cancelUpload}
          onComplete={() => { setUploadId(null); setFiles([]); setStatus({ type: "success", message: "Upload completato." }); }}
          onError={m => { setUploadId(null); setStatus({ type: "error", message: m }); }}
        />
      )}

      <StatusToast status={status} onDismiss={() => setStatus({ type: "", message: "" })} autoHideMs={4000} />

      {/* ── CREATE MODAL ── */}
      <Modal open={createOpen} onClose={closeCreate}
        title="Nuova collezione" description="Il nome viene convertito in maiuscolo automaticamente.">
        <div className="space-y-4">
          <div>
            <label className="block text-xs text-zinc-500 mb-1.5 uppercase tracking-widest">Nome</label>
            <input
              value={newCollectionName}
              onChange={e => setNewCollectionName(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === "Enter" && createCollection()}
              placeholder="es. MY_DOCS"
              className="w-full px-3.5 py-2.5 bg-[#0a0a0a] border border-zinc-800 rounded-xl text-sm text-zinc-100 placeholder:text-zinc-700 focus:outline-none focus:border-yellow-400/50 focus:ring-1 focus:ring-yellow-400/20 transition font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-zinc-500 mb-1.5 uppercase tracking-widest">Descrizione (facoltativa)</label>
            <input
              value={newCollectionDescription}
              onChange={e => setNewCollectionDescription(e.target.value)}
              placeholder="Breve descrizione..."
              className="w-full px-3.5 py-2.5 bg-[#0a0a0a] border border-zinc-800 rounded-xl text-sm text-zinc-100 placeholder:text-zinc-700 focus:outline-none focus:border-yellow-400/50 focus:ring-1 focus:ring-yellow-400/20 transition"
            />
          </div>
          <div className="flex justify-end gap-2 pt-1">
            <Btn variant="ghost" onClick={closeCreate}>Annulla</Btn>
            <Btn variant="primary" onClick={createCollection}><Plus className="w-3.5 h-3.5" /> Crea</Btn>
          </div>
        </div>
      </Modal>

      {/* ── UPLOAD MODAL ── */}
      <Modal open={uploadOpen} onClose={() => { setUploadOpen(false); setFiles([]); }}
        title="Carica documenti"
        description="PDF, TXT, RTF, JPG/PNG/TIFF/BMP — OCR multilingua automatico."
        width="max-w-2xl">
        <div className="space-y-5">
          <div>
            <label className="block text-xs text-zinc-500 mb-1.5 uppercase tracking-widest">Collezione</label>
            <input
              list="coll-list"
              value={selectedCollection}
              onChange={e => setSelectedCollection(e.target.value)}
              placeholder="Seleziona o digita il nome..."
              className="w-full px-3.5 py-2.5 bg-[#0a0a0a] border border-zinc-800 rounded-xl text-sm text-zinc-100 placeholder:text-zinc-700 focus:outline-none focus:border-yellow-400/50 focus:ring-1 focus:ring-yellow-400/20 transition"
            />
            <datalist id="coll-list">{normalized.map(c => <option key={c.name} value={c.name} />)}</datalist>
          </div>

          <div>
            <label className="block text-xs text-zinc-500 mb-1.5 uppercase tracking-widest">File</label>
            <input ref={fileInputRef} id="fi" type="file" multiple onChange={handleFiles}
              accept=".pdf,.txt,.rtf,.jpg,.jpeg,.png,.tiff,.tif,.bmp" className="hidden" />
            <label htmlFor="fi"
              className="flex flex-col items-center justify-center gap-2 border-2 border-dashed border-zinc-800 hover:border-yellow-400/40 rounded-xl py-8 cursor-pointer transition group bg-[#0a0a0a] hover:bg-yellow-400/[0.03]">
              <div className="w-10 h-10 rounded-xl bg-zinc-800 group-hover:bg-yellow-400/10 flex items-center justify-center transition">
                <Upload className="w-5 h-5 text-zinc-500 group-hover:text-yellow-400 transition" />
              </div>
              <div className="text-center">
                <p className="text-sm text-zinc-400 group-hover:text-zinc-200 transition">
                  {files.length === 0 ? "Clicca per scegliere i file" : `${files.length} file selezionati`}
                </p>
                <p className="text-xs text-zinc-700 mt-0.5">PDF, TXT, RTF, immagini</p>
              </div>
            </label>
          </div>

          {files.length > 0 && (
            <div className="border border-zinc-800 rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-3.5 py-2 border-b border-zinc-800 bg-[#0d0d0d]">
                <span className="text-xs text-zinc-500">{files.length} file in coda</span>
                <button onClick={() => setFiles([])} className="text-xs text-zinc-600 hover:text-zinc-300 transition">Svuota</button>
              </div>
              <ul className="max-h-48 overflow-auto divide-y divide-zinc-800/50">
                {files.map((f, i) => (
                  <li key={`${f.name}-${i}`} className="flex items-center gap-3 px-3.5 py-2 hover:bg-zinc-800/20 transition group">
                    <FileText className="w-3.5 h-3.5 text-zinc-600 shrink-0" />
                    <span className="text-xs text-zinc-300 truncate flex-1 font-mono">{f.name}</span>
                    <span className="text-xs text-zinc-600 shrink-0">{fmt(f.size)}</span>
                    <button onClick={() => setFiles(p => p.filter((_, j) => j !== i))}
                      className="opacity-0 group-hover:opacity-100 text-zinc-600 hover:text-red-400 transition">
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex justify-end gap-2 pt-1 border-t border-zinc-800/60">
            <Btn variant="ghost" onClick={() => { setUploadOpen(false); setFiles([]); }}>Chiudi</Btn>
            <Btn variant="primary" onClick={startUpload} disabled={!selectedCollection || files.length === 0}>
              <Upload className="w-3.5 h-3.5" /> Avvia upload
            </Btn>
          </div>
        </div>
      </Modal>

      {/* ══════════════════════════════════════════════════════
          MAIN LAYOUT  –  Sidebar + Content
      ══════════════════════════════════════════════════════ */}
      <div className="flex min-h-screen bg-[#0a0a0a] text-zinc-100">

        {/* ── SIDEBAR ── */}
        <aside className="w-60 shrink-0 border-r border-zinc-900 flex flex-col bg-[#0d0d0d] sticky top-0 h-screen">
          <div className="px-5 py-6 border-b border-zinc-900">
            <img src="/logo_completo.png" alt="Logo" className="h-6 w-auto opacity-90" />
          </div>

          <nav className="flex-1 p-3 space-y-0.5">
            <div className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-yellow-400/10 text-yellow-400 cursor-default">
              <Layers className="w-4 h-4 shrink-0" />
              <span className="text-sm font-medium">Collezioni</span>
              <span className="ml-auto text-xs font-bold tabular-nums bg-yellow-400/20 px-1.5 py-0.5 rounded-md">
                {normalized.length}
              </span>
            </div>
          </nav>

          {/*<div className="p-4 border-t border-zinc-900 space-y-3">*/}
          {/*  <p className="text-[10px] text-zinc-600 uppercase tracking-widest">Stato sistema</p>*/}
          {/*  <div className="flex items-center gap-2.5">*/}
          {/*    <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />*/}
          {/*    <span className="text-xs text-zinc-400">Weaviate connesso</span>*/}
          {/*  </div>*/}
          {/*  <div className="flex items-center gap-2.5">*/}
          {/*    <Activity className="w-3.5 h-3.5 text-zinc-600" />*/}
          {/*    <span className="text-xs text-zinc-500 font-mono">{config.weaviateHost}:{config.weaviatePort}</span>*/}
          {/*  </div>*/}
          {/*</div>*/}
        </aside>

        {/* ── MAIN CONTENT ── */}
        <div className="flex-1 flex flex-col min-w-0">

          {/* Top bar */}
          <header className="flex items-center justify-between gap-4 px-8 py-4 border-b border-zinc-900 bg-[#0d0d0d] sticky top-0 z-10">
            <div>
              <h1 className="text-base font-semibold text-zinc-100">Document Manager</h1>
              <p className="text-xs text-zinc-600">Gestione e indicizzazione dei docu sul database vettoriale</p>
            </div>
            <div className="flex items-center gap-2">
              <Btn variant="ghost" onClick={openCreate}>
                <Plus className="w-3.5 h-3.5 text-yellow-400" /> Nuova collezione
              </Btn>
              <Btn variant="primary" onClick={() => openUpload()} disabled={normalized.length === 0}>
                <Upload className="w-3.5 h-3.5" /> Carica documenti
              </Btn>
            </div>
          </header>

          {/* Content */}
          <main className="flex-1 p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="relative flex-1 max-w-xs">
                <Search className="w-3.5 h-3.5 text-zinc-600 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                <input
                  value={collectionQuery}
                  onChange={e => setCollectionQuery(e.target.value)}
                  placeholder="Cerca collezione..."
                  className="w-full pl-9 pr-3 py-2 text-sm bg-[#111] border border-zinc-800 rounded-xl text-zinc-200 placeholder:text-zinc-700 focus:outline-none focus:border-yellow-400/40 focus:ring-1 focus:ring-yellow-400/20 transition"
                />
              </div>
              {collectionQuery && (
                <span className="text-xs text-zinc-500">{visible.length} risultati</span>
              )}
            </div>

            {visible.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24 gap-4">
                <div className="w-16 h-16 rounded-2xl border border-zinc-800 bg-zinc-900/40 flex items-center justify-center">
                  <FolderOpen className="w-7 h-7 text-zinc-700" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-zinc-400">
                    {normalized.length === 0 ? "Nessuna collezione" : "Nessun risultato"}
                  </p>
                  <p className="text-xs text-zinc-600 mt-1">
                    {normalized.length === 0 ? "Inizia creando la tua prima collezione." : "Prova con un termine diverso."}
                  </p>
                </div>
                {normalized.length === 0 && (
                  <Btn variant="ghost" onClick={openCreate} className="mt-2">
                    <Plus className="w-3.5 h-3.5 text-yellow-400" /> Crea collezione
                  </Btn>
                )}
              </div>
            ) : (
              <div className="border border-zinc-800 rounded-2xl overflow-hidden">
                {/* Table header */}
                <div className="grid grid-cols-[1fr_auto_auto] bg-[#111] border-b border-zinc-800">
                  <div className="px-5 py-3 text-[10px] text-zinc-600 uppercase tracking-widest font-medium">Nome</div>
                  {/*<div className="px-5 py-3 text-[10px] text-zinc-600 uppercase tracking-widest font-medium text-right">Stato</div>*/}
                  <div className="px-5 py-3 text-[10px] text-zinc-600 uppercase tracking-widest font-medium text-center w-40">Azioni</div>
                </div>

                {/* Table rows */}
                <div className="divide-y divide-zinc-900">
                  {visible.map((c) => (
                    <div
                      key={c.name}
                      className={`grid grid-cols-[1fr_auto_auto] items-center transition-colors ${hoveredRow === c.name ? "bg-zinc-900/60" : "bg-[#0d0d0d]"}`}
                      onMouseEnter={() => setHoveredRow(c.name)}
                      onMouseLeave={() => setHoveredRow(null)}
                    >
                      <div className="px-5 py-4 flex items-center gap-3 min-w-0">
                        <div className="w-8 h-8 rounded-lg bg-yellow-400/10 border border-yellow-400/10 flex items-center justify-center shrink-0">
                          <Database className="w-4 h-4 text-yellow-400/80" />
                        </div>
                        <div className="min-w-0">
                          <p className="text-sm font-semibold text-zinc-100 font-mono truncate">{c.name}</p>
                          {c.description && (
                            <p className="text-xs text-zinc-600 truncate">{c.description}</p>
                          )}
                        </div>
                      </div>

                      {/*<div className="px-5 py-4">*/}
                      {/*  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-green-950/50 border border-green-800/40 text-xs text-green-400">*/}
                      {/*    <span className="w-1.5 h-1.5 rounded-full bg-green-400 inline-block" />*/}
                      {/*    Attiva*/}
                      {/*  </span>*/}
                      {/*</div>*/}

                      <div className="px-5 py-4 flex items-center gap-2 justify-end w-40">
                        <Btn variant="primary" className="text-xs px-3 py-1.5" onClick={() => openUpload(c.name)}>
                          <Upload className="w-3 h-3" /> Carica
                        </Btn>
                        <button
                          onClick={() => deleteCollection(c.name)}
                          className={`w-8 h-8 flex items-center justify-center rounded-lg border border-zinc-600 text-zinc-600 hover:text-red-400 hover:border-red-800/60 hover:bg-red-950/30 opacity-100`}
                          aria-label={`Elimina ${c.name}`}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Table footer */}
                {/*<div className="px-5 py-3 border-t border-zinc-900 bg-[#0d0d0d] flex items-center justify-between">*/}
                {/*  <span className="text-xs text-zinc-600">{visible.length} collezion{visible.length !== 1 ? "i" : "e"}</span>*/}
                {/*  <button onClick={() => void loadCollections()}*/}
                {/*    className="text-xs text-zinc-600 hover:text-zinc-300 flex items-center gap-1.5 transition">*/}
                {/*    <ChevronRight className="w-3 h-3" /> Aggiorna*/}
                {/*  </button>*/}
                {/*</div>*/}
              </div>
            )}
          </main>
        </div>
      </div>
    </>
  );
};

export default WeaviateDocumentManager;
