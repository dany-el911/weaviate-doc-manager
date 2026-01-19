import React, { useState, useEffect, useRef } from "react";
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
} from "lucide-react";

// --- INTERFACCE ---

interface WeaviateClass {
    class?: string;
    name?: string;
}

interface StatusState {
    type: "error" | "success" | "loading" | "";
    message: string;
}

type IngestMode = "precision" | "balanced" | "long_context";

interface ConfigState {
    weaviateHost: string;
    weaviatePort: string;
    ollamaUrl: string;
    embedModel: string;
    ingestMode: IngestMode;
}

interface ProgressData {
    stage: string;
    current: number;
    total: number;
    percent: number;
    error?: string;
    currentFile?: string;
}

// --- COMPONENTE LOADER ISOLATO (Smart Component) ---
// Ora gestisce lui il polling, così il padre non deve re-renderizzare

interface UploadLoaderProps {
    uploadId: string;
    onCancel: () => void;
    onComplete: () => void;
    onError: (msg: string) => void;
}

const UploadLoader: React.FC<UploadLoaderProps> = ({ uploadId, onCancel, onComplete, onError }) => {
    const [progress, setProgress] = useState<ProgressData>({
        stage: "starting",
        current: 0,
        total: 0,
        percent: 0,
    });

    // Polling logic spostata qui dentro
    useEffect(() => {
        let isMounted = true;

        const interval = setInterval(async () => {
            try {
                const res = await fetch(`http://127.0.0.1:5001/api/upload-progress/${uploadId}`);
                if (!res.ok) return;

                const data: ProgressData = await res.json();

                if (isMounted) {
                    setProgress(data);
                }

                if (data.stage === "done") {
                    clearInterval(interval);
                    // Piccolo ritardo per mostrare il 100% prima di chiudere
                    setTimeout(() => {
                        if (isMounted) onComplete();
                    }, 500);
                }

                if (data.stage === "error") {
                    clearInterval(interval);
                    if (isMounted) onError(data.error || "Errore sconosciuto durante l'upload");
                }

            } catch (err) {
                console.error("Polling error:", err);
            }
        }, 200);

        return () => {
            isMounted = false;
            clearInterval(interval);
        };
    }, [uploadId, onComplete, onError]);

    const getStageLabel = (): string => {
        if (progress.stage === "starting") return "Inizializzazione...";
        if (progress.stage === "processing") return "Elaborazione documenti";
        if (progress.stage === "done") return "Completato!";
        return "Caricamento in corso...";
    };

    return (
        <div className="fixed inset-0 bg-black/90 flex flex-col items-center justify-center z-50">
            <div className="rounded-xl bg-[#1a1a1a] p-8 shadow-2xl border border-zinc-700 text-center w-[400px]">
                <svg
                    className="animate-spin h-12 w-12 text-yellow-400 mx-auto mb-6"
                    viewBox="0 0 50 50"
                >
                    <circle
                        cx="25"
                        cy="25"
                        r="20"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="5"
                        strokeDasharray="90"
                        strokeLinecap="round"
                    />
                </svg>

                <p className="text-yellow-400 font-semibold text-xl mb-1">
                    {progress.percent}%
                </p>

                <p className="text-gray-300 text-base mb-1">
                    {getStageLabel()}
                </p>

                <p className="text-gray-400 text-sm mb-2">
                    File {progress.current} di {progress.total}
                </p>

                {progress.currentFile && (
                    <p className="text-gray-500 text-xs mb-4 truncate px-2">
                        {progress.currentFile}
                    </p>
                )}

                <div className="w-full bg-zinc-700 rounded-full h-3 overflow-hidden mb-4 border border-zinc-600">
                    <div
                        className="h-full bg-yellow-400 transition-all duration-300 ease-out"
                        style={{ width: `${progress.percent}%` }}
                    />
                </div>

                <button
                    type="button"
                    onClick={onCancel}
                    className="mt-2 w-full py-2 rounded-lg border border-red-500/50 text-red-400 hover:bg-red-900/30 hover:text-red-200 transition text-sm font-medium"
                >
                    Annulla caricamento
                </button>
            </div>
        </div>
    );
};


// --- MAIN COMPONENT ---

const WeaviateDocumentManager: React.FC = () => {
    const [collections, setCollections] = useState<WeaviateClass[]>([]);
    const [selectedCollection, setSelectedCollection] = useState<string>("");
    const [newCollectionName, setNewCollectionName] = useState<string>("");
    const [files, setFiles] = useState<File[]>([]);

    // Stato ridotto al minimo: sappiamo solo SE stiamo caricando e QUALE ID
    const [uploadId, setUploadId] = useState<string | null>(null);

    const [status, setStatus] = useState<StatusState>({
        type: "",
        message: "",
    });

    const [config, setConfig] = useState<ConfigState>({
        weaviateHost: "127.0.0.1",
        weaviatePort: "8080",
        ollamaUrl: "http://127.0.0.1:11434",
        embedModel: "qwen3-embedding:4b", // mxbai-embed-large
        ingestMode: "balanced",
    });

    const uploadAbortControllerRef = useRef<AbortController | null>(null);

    useEffect(() => {
        void loadCollections();
    }, []);

    const loadCollections = async (): Promise<void> => {
        try {
            const response = await fetch(
                `http://${config.weaviateHost}:${config.weaviatePort}/v1/schema`
            );

            if (!response.ok) throw new Error("Failed to fetch schema");

            const data = (await response.json()) as { classes?: WeaviateClass[] };
            const filtered = (data.classes ?? []).filter((col: WeaviateClass) => {
                const name = col.class || col.name || "";
                return !name.startsWith("ELYSIA_") && !name.startsWith("_");
            });

            setCollections(filtered);
        } catch (error) {
            console.error("Error loading collections:", error);
            setStatus({ type: "error", message: "Impossibile connettersi a Weaviate" });
        }
    };

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>): void => {
        if (!e.target.files) return;
        const selected = Array.from(e.target.files);
        setFiles((prev) => {
            const existingKeys = new Set(prev.map((f) => `${f.name}::${f.size}`));
            const merged = [...prev];
            for (const file of selected) {
                if (!existingKeys.has(`${file.name}::${file.size}`)) {
                    merged.push(file);
                }
            }
            return merged;
        });
        e.target.value = "";
    };

    const removeFile = (index: number): void => {
        setFiles((prev) => prev.filter((_, i) => i !== index));
    };

    const createCollection = async (): Promise<void> => {
        if (!newCollectionName.trim()) {
            setStatus({ type: "error", message: "Inserisci un nome" });
            return;
        }
        setStatus({ type: "loading", message: "Creazione in corso..." });

        try {
            const response = await fetch("http://localhost:5001/api/create-collection", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: newCollectionName, config }),
            });
            if (!response.ok) throw new Error("Errore creazione");

            setStatus({ type: "success", message: `Collezione "${newCollectionName}" creata` });
            setNewCollectionName("");
            void loadCollections();
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Errore sconosciuto";
            setStatus({ type: "error", message });
        }
    };

    const deleteCollection = async (name: string): Promise<void> => {
        if (!confirm(`Vuoi eliminare "${name}"?`)) return;
        try {
            const res = await fetch("http://127.0.0.1:5001/api/delete-collection", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name }),
            });
            if (!res.ok) throw new Error("Errore eliminazione");
            setStatus({ type: "success", message: `Eliminata "${name}"` });
            void loadCollections();
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Errore sconosciuto";
            setStatus({ type: "error", message });
        }
    };

    const startUpload = async (): Promise<void> => {
        if (!selectedCollection || files.length === 0) return;

        const newUploadId = crypto.randomUUID
            ? crypto.randomUUID()
            : `up_${Date.now()}`;

        // 1. Impostiamo l'ID: questo farà apparire il Loader
        setUploadId(newUploadId);

        // Controller per l'upload iniziale dei file (la POST)
        const controller = new AbortController();
        uploadAbortControllerRef.current = controller;

        const formData = new FormData();
        formData.append("collection", selectedCollection);
        formData.append("config", JSON.stringify(config));
        formData.append("uploadId", newUploadId);
        files.forEach((file) => formData.append("files", file));

        // 2. Lanciamo la richiesta "Fire and Forget" (o quasi)
        // Non aspettiamo la risposta qui per aggiornare l'UI,
        // lasciamo che sia il Loader a fare polling.
        // Tuttavia dobbiamo gestire gli errori di rete iniziali della POST.

        fetch("http://127.0.0.1:5001/api/upload-documents", {
            method: "POST",
            body: formData,
            signal: controller.signal,
        })
            .then(async (response) => {
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || "Errore upload");
                }
                // Se va a buon fine, il polling nel Loader se ne accorgerà (state="done")
                // quindi qui non dobbiamo fare nulla di specifico sull'UI.
            })
            .catch((error) => {
                if (error.name === 'AbortError') return;
                // Se la POST fallisce subito (es. server down), chiudiamo il loader
                setUploadId(null);
                setStatus({ type: "error", message: error.message });
            });
    };

    // Callback chiamata dal Loader quando ha finito
    const handleUploadComplete = () => {
        setUploadId(null);
        setFiles([]); // Pulisce i file
        setSelectedCollection(""); // <--- AGGIUNGI QUESTA RIGA: Resetta la dropdown
        setStatus({
            type: "success",
            message: `Upload completato con successo.` // Messaggio generico dato che la collezione è deselezionata
        });
    };

    // Callback chiamata dal Loader se rileva un errore dal server
    const handleUploadError = (msg: string) => {
        setUploadId(null);
        setStatus({ type: "error", message: msg });
    };

    // Callback per annullamento manuale
    const cancelUpload = async (): Promise<void> => {
        // Interrompiamo la POST se è ancora in corso
        if (uploadAbortControllerRef.current) {
            uploadAbortControllerRef.current.abort();
        }

        const currentId = uploadId;
        // Chiudiamo subito il loader per reattività UI
        setUploadId(null);

        // Chiamata backend per pulizia
        if (currentId) {
            try {
                await fetch("http://127.0.0.1:5001/api/cancel-upload", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ uploadId: currentId }),
                });
            } catch (e) { console.error(e); }
        }

        setStatus({ type: "", message: "" });
    };

    // --- RENDER HELPERS ---

    const StatusAlert: React.FC = () => {
        if (!status.message) return null;
        const styles = {
            error: "bg-red-900/20 border-red-500 text-red-300",
            success: "bg-green-900/20 border-green-500 text-green-300",
            loading: "bg-blue-900/20 border-blue-500 text-blue-300",
            "": "",
        };
        const icons = {
            error: <AlertCircle className="w-5 h-5" />,
            success: <CheckCircle className="w-5 h-5" />,
            loading: <Loader className="w-5 h-5 animate-spin" />,
            "": null,
        };
        return (
            <div className={`flex items-center gap-3 p-4 rounded-lg border ${styles[status.type]} mb-6`}>
                {icons[status.type]}
                <span>{status.message}</span>
            </div>
        );
    };

    return (
        <>
            {/* Il Loader appare solo se c'è un uploadId.
                Tutta la logica di aggiornamento (re-render rapido) avviene SOLO dentro <UploadLoader> */}
            {uploadId && (
                <UploadLoader
                    uploadId={uploadId}
                    onCancel={cancelUpload}
                    onComplete={handleUploadComplete}
                    onError={handleUploadError}
                />
            )}

            <div className="min-h-screen bg-[#0f0f0f] text-gray-100 p-6">
                <div className="max-w-6xl mx-auto">
                    <StatusAlert />

                    {/* SEZIONE CREAZIONE */}
                    <div className="bg-[#1a1a1a] border border-zinc-800 rounded-xl shadow-md p-6 flex flex-col min-h-[260px] max-h-[410px]">
                        <h2 className="text-xl font-semibold text-yellow-400 mb-4 flex items-center gap-2">
                            <Plus className="w-5 h-5" /> Crea Nuova Collezione
                        </h2>
                        <input
                            type="text"
                            value={newCollectionName}
                            placeholder="es. MY_DOCS"
                            onChange={(e) => setNewCollectionName(e.target.value.toUpperCase())}
                            className="w-full px-4 py-2 bg-[#0f0f0f] border border-zinc-700 rounded-lg text-gray-200 focus:border-yellow-400 focus:outline-none transition"
                        />
                        <button
                            onClick={createCollection}
                            className="mt-4 w-full bg-yellow-400 text-black font-medium py-3 rounded-lg hover:bg-yellow-500 transition"
                        >
                            + Crea Collezione
                        </button>

                        <div className="mt-6 space-y-2 flex-1 overflow-y-auto scrollbar-thin" style={{ scrollbarColor: "#facc15 #0b0b0b" }}>
                            <h3 className="text-sm text-gray-300 mb-2 flex items-center gap-2">
                                <Database className="w-4 h-4 text-yellow-400" /> Collezioni esistenti
                            </h3>
                            {collections.length === 0 ? (
                                <p className="text-gray-500 italic text-sm">Nessuna collezione trovata</p>
                            ) : (
                                collections.map((col, idx) => (
                                    <div key={idx} className="px-3 py-2 rounded bg-[#0f0f0f] border border-zinc-700 text-gray-300 text-sm flex justify-between items-center hover:border-zinc-500 transition">
                                        <span>{col.class || col.name}</span>
                                        <button onClick={() => deleteCollection((col.class || col.name)!)} className="text-red-400 hover:text-red-300 p-1">
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* SEZIONE UPLOAD */}
                    <div className="bg-[#1a1a1a] border border-zinc-800 rounded-xl shadow-md p-6 mt-6">
                        <h2 className="text-xl font-semibold text-yellow-400 mb-4 flex items-center gap-2">
                            <Upload className="w-5 h-5" /> Carica Documenti
                        </h2>

                        <div className="grid md:grid-cols-2 gap-6">
                            <div>
                                <label className="block text-sm text-gray-300 mb-1">Seleziona Collezione</label>
                                <select
                                    value={selectedCollection}
                                    onChange={(e) => setSelectedCollection(e.target.value)}
                                    className="
                                        w-full
                                        appearance-none
                                        px-4
                                        py-2
                                        bg-[#0f0f0f]
                                        border
                                        border-zinc-700
                                        rounded-lg
                                        text-gray-200
                                        focus:border-yellow-400
                                        focus:outline-none
                                        focus:ring-2
                                        focus:ring-yellow-400
                                        bg-no-repeat
                                        pr-10
                                    "
                                    style={{
                                        backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23facc15'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7' /%3E%3C/svg%3E")`,
                                        backgroundPosition: "right 0.75rem center",
                                        backgroundSize: "1rem",
                                    }}
                                >
                                    <option value="">-- Scegli collezione --</option>

                                    {collections.map((col, idx) => (
                                        <option key={idx} value={col.class || col.name}>
                                            {col.class || col.name}
                                        </option>
                                    ))}
                                </select>

                            </div>
                            <div>
                                <label className="block text-sm text-gray-300 mb-1">Seleziona File (PDF/TXT/RTF/Immagini)</label>
                                <input
                                    type="file"
                                    multiple
                                    accept=".pdf,.txt,.rtf,.jpg,.jpeg,.png,.tiff,.tif,.bmp"
                                    onChange={handleFileChange}
                                    className="w-full px-4 py-2 bg-[#0f0f0f] border border-zinc-700 rounded-lg text-gray-200 focus:border-yellow-400 focus:outline-none"
                                />
                                <p className="text-xs text-gray-500 mt-1">
                                    📸 OCR automatico multilingua per PDF scansionati e immagini
                                </p>
                            </div>
                        </div>

                        <div className="space-y-1">
                            <label className="block text-sm font-medium">
                                Modalità di ingest
                            </label>

                            <select
                                className="
                                    w-full
                                    appearance-none
                                    border
                                    rounded
                                    px-3
                                    py-2
                                    text-sm
                                    bg-zinc-900
                                    text-zinc-100
                                    border-zinc-700
                                    focus:outline-none
                                    focus:ring-2
                                    focus:ring-yellow-400
                                    bg-no-repeat
                                    bg-right
                                    bg-[length:1rem]
                                    pr-10
                                "
                                style={{
                                    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23facc15'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7' /%3E%3C/svg%3E")`,
                                    backgroundPosition: "right 0.75rem center",
                                }}
                                value={config.ingestMode}
                                onChange={(e) =>
                                    setConfig({
                                        ...config,
                                        ingestMode: e.target.value as IngestMode,
                                    })
                                }
                            >
                                <option value="precision">
                                    Precisione — più frammenti, maggiore accuratezza
                                </option>
                                <option value="balanced">
                                    Bilanciata — consigliata
                                </option>
                                <option value="long_context">
                                    Contesto lungo — documenti narrativi
                                </option>
                            </select>



                            <p className="text-xs text-gray-500">
                                La modalità determina come il testo viene suddiviso e indicizzato.
                            </p>
                        </div>


                        {/*<div className="mt-6 border border-zinc-700 rounded-lg p-4 bg-[#0b0b0b]">*/}
                        {/*    <div className="flex items-center gap-2 mb-2">*/}
                        {/*        <AlertCircle className="w-4 h-4 text-yellow-400" />*/}
                        {/*        /!* Etichetta più chiara e professionale *!/*/}
                        {/*        <span className="text-m font-medium text-gray-200">Dimensione Frammenti (Chunk Size)</span>*/}
                        {/*    </div>*/}

                        {/*    <input*/}
                        {/*        type="number"*/}
                        {/*        min={64}*/}
                        {/*        max={2048}*/}
                        {/*        step={64}*/}
                        {/*        value={config.chunkSize}*/}
                        {/*        onChange={(e) => setConfig({ ...config, chunkSize: e.target.value })}*/}
                        {/*        className="w-full px-3 py-2 bg-[#0f0f0f] border border-zinc-700 rounded-lg text-gray-200 text-sm focus:border-yellow-400 focus:outline-none"*/}
                        {/*    />*/}

                        {/*    /!* Spiegazione utile invece di un avvertimento generico *!/*/}
                        {/*    <p className="text-xs text-gray-400 mt-2 leading-relaxed">*/}
                        {/*        Definisce la lunghezza massima di ogni blocco di testo.*/}
                        {/*        <span className="block mt-1 text-amber-400/90">*/}
                        {/*        Consigliato: 1024. Un valore più basso aumenta la precisione per dati specifici, un valore più alto mantiene più contesto narrativo.*/}
                        {/*    </span>*/}
                        {/*    </p>*/}
                        {/*</div>*/}

                        {files.length > 0 && (
                            <div className="mt-4 p-4 bg-[#0f0f0f] border border-zinc-700 rounded-lg">
                                <h3 className="text-sm mb-2 text-gray-300 flex items-center gap-2">
                                    <FileText className="w-4 h-4 text-yellow-400" /> File Selezionati ({files.length})
                                </h3>
                                <ul className="space-y-1 text-sm max-h-40 overflow-y-auto pr-2 scrollbar-thin" style={{ scrollbarColor: "#facc15 #0b0b0b" }}>
                                    {files.map((file, idx) => (
                                        <li key={idx} className="flex justify-between items-center text-gray-400 bg-[#141414] px-2 py-1 rounded">
                                            <span className="truncate flex-1">{file.name}</span>
                                            <button onClick={() => removeFile(idx)} className="text-red-400 ml-2 hover:text-red-300">
                                                <Trash2 className="w-4 h-4" />
                                            </button>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        <button
                            disabled={!!uploadId || !selectedCollection || files.length === 0}
                            onClick={startUpload}
                            className="mt-6 w-full bg-yellow-400 text-black font-medium py-3 rounded-lg hover:bg-yellow-500 disabled:bg-zinc-700 disabled:text-zinc-500 transition flex items-center justify-center gap-2"
                        >
                            {uploadId ? <Loader className="animate-spin w-5 h-5"/> : <Upload className="w-5 h-5"/>}
                            {uploadId ? "Avvio..." : "Carica Documenti"}
                        </button>
                    </div>
                </div>
            </div>
        </>
    );
};

export default WeaviateDocumentManager;