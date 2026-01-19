from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import requests
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import Filter

from typing import Dict, Any, List
import tempfile
import os
import uuid
import json

from langdetect import detect

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions

# Tokenizer
from transformers import AutoTokenizer

app = Flask(__name__)
CORS(app)

# Dizionario globale per tracciare il progresso degli upload
upload_progress: Dict[str, Dict[str, Any]] = {}

# Traccia gli upload per poterli cancellare:
# upload_id -> { collection, weaviateHost, weaviatePort, file_ids }
UPLOAD_SESSIONS: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# CONFIG / PRESETS (allineati alla versione vecchia funzionante)
# -----------------------------
INGEST_PRESETS = {
    "precision":    {"max_tokens": 384,  "overlap_tokens": 64},
    "balanced":     {"max_tokens": 640,  "overlap_tokens": 80},
    "long_context": {"max_tokens": 1024, "overlap_tokens": 96},
}

_TOKENIZER_CACHE: Dict[str, Any] = {}

# Assicurati che il processo Python veda Tesseract
os.environ["PATH"] = "/usr/local/bin:" + os.environ.get("PATH", "")

# (Opzionale ma consigliato se non l'hai settato in shell)
# os.environ.setdefault("TESSDATA_PREFIX", "/usr/local/share/tessdata/")

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)

# Docling converter globale configurato per PDF scansione (OCR)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


# -----------------------------
# Helpers
# -----------------------------
def normalize_text(text: str) -> str:
    """
    Rimuove caratteri Unicode non validi per UTF-8 (es. surrogates),
    in modo da evitare errori 'surrogates not allowed'.
    """
    return (text or "").encode("utf-8", "ignore").decode("utf-8")


def init_weaviate_client(host: str, port: str):
    """Inizializza client Weaviate"""
    return weaviate.connect_to_local(
        host=host,
        port=int(port),
        grpc_port=50051,
    )


def _resolve_tokenizer_name(embed_model: str) -> str:
    """
    Mappa il nome del modello embedding (Ollama) a un tokenizer HuggingFace.
    """
    m = (embed_model or "").lower()

    if "qwen3-embedding" in m:
        return "Qwen/Qwen3-Embedding-4B"

    if "mxbai-embed-large" in m:
        return "mixedbread-ai/mxbai-embed-large-v1"

    # fallback generico
    return "gpt2"


def get_tokenizer(embed_model: str):
    if embed_model in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[embed_model]

    tokenizer_name = _resolve_tokenizer_name(embed_model)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    _TOKENIZER_CACHE[embed_model] = tok
    return tok


def chunk_text_token_based(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """
    Chunking a token + overlap fisso in token.
    Strategia:
    - split semantico per paragrafi
    - packing entro max_tokens
    - overlap fisso tra chunk
    """
    text = normalize_text(text).strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks_token_ids: List[List[int]] = []
    current: List[int] = []

    def flush():
        nonlocal current
        if current:
            chunks_token_ids.append(current)
            current = []

    for p in paragraphs:
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        if not p_ids:
            continue

        # paragrafo più lungo di max_tokens -> spezza direttamente
        if len(p_ids) > max_tokens:
            flush()
            start = 0
            while start < len(p_ids):
                end = start + max_tokens
                chunks_token_ids.append(p_ids[start:end])
                start = end
            continue

        # se aggiungerlo sfora -> chiudi chunk
        if len(current) + len(p_ids) > max_tokens:
            flush()

        current.extend(p_ids)

    flush()

    if not chunks_token_ids:
        return []

    # Overlap fisso
    out: List[str] = []
    prev_tail: List[int] = []

    for i, ids in enumerate(chunks_token_ids):
        if i == 0 or overlap_tokens <= 0:
            merged = ids
        else:
            merged = prev_tail + ids
            if len(merged) > max_tokens:
                merged = merged[-max_tokens:]

        out.append(tokenizer.decode(merged))
        prev_tail = ids[-overlap_tokens:] if overlap_tokens > 0 else []

    # pulizia finale
    return [normalize_text(c).strip() for c in out if c and c.strip()]


def get_ollama_embedding(text: str, ollama_url: str, model: str) -> list[float]:
    """
    Genera embedding tramite Ollama.
    """
    print(f"EMBEDDING REQUEST: url={ollama_url}, model={model}, text_len={len(text)}")
    payload = {"model": model, "prompt": text}
    try:
        resp = requests.post(f"{ollama_url}/api/embeddings", json=payload, timeout=120)
        print(f"EMBEDDING RESPONSE: status={resp.status_code}")
        if not resp.ok:
            raise Exception(f"Ollama error: {resp.status_code} - {resp.text[:200]}")
        data = resp.json()
        print(f"EMBEDDING OK: len={len(data['embedding'])}")
        return data["embedding"]
    except Exception as e:
        print(f"EMBEDDING ERROR: {e}")
        raise


def extract_with_docling(
        file_path: str,
        tokenizer,
        max_tokens: int = 1024,
        overlap_tokens: int = 128
) -> List[Dict[str, Any]]:
    """
    Estrae contenuto usando Docling con chunking intelligente.
    APPROCCIO SEMPLIFICATO:
    1. Docling converte il documento
    2. Estrae il testo pulito (senza formattazione markdown)
    3. Applica chunking token-based con overlap

    Args:
        file_path: Percorso del file da processare
        tokenizer: Tokenizer per chunking token-based
        max_tokens: Massimo numero di token per chunk (default: 1024)
        overlap_tokens: Token di overlap tra chunk consecutivi (default: 128)

    Returns:
        Lista di blocchi con struttura: {page, kind, content}
    """
    try:
        print(f"[Docling] Processing: {file_path}")

        # Gestione file TXT: Docling non supporta .txt direttamente
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_text = f.read()
            print(f"[Docling] TXT file read directly: {len(full_text)} chars")
        else:
            # Converti documento con Docling
            result = doc_converter.convert(file_path)
            doc = result.document

            # Prova diversi metodi per estrarre il testo pulito
            full_text = ""

            # Metodo 1: export_to_text (più pulito, senza markdown)
            if hasattr(doc, 'export_to_text'):
                try:
                    full_text = doc.export_to_text()
                    print(f"[Docling] Used export_to_text: {len(full_text)} chars")
                except Exception as e:
                    print(f"[Docling] export_to_text failed: {e}")

            # Metodo 2: iterate_items se export_to_text non disponibile o vuoto
            if not full_text or len(full_text.strip()) < 10:
                try:
                    text_parts = []
                    for item in doc.iterate_items():
                        # Gestisci sia tuple (item, level) che oggetti singoli
                        if isinstance(item, tuple):
                            item = item[0]
                        if hasattr(item, 'text') and item.text:
                            clean_text = normalize_text(item.text).strip()
                            if clean_text:
                                text_parts.append(clean_text)
                    full_text = "\n\n".join(text_parts)
                    print(f"[Docling] Used iterate_items: {len(text_parts)} elements, {len(full_text)} chars")
                except Exception as e:
                    print(f"[Docling] iterate_items failed: {e}")

            # Metodo 3: Fallback a export_to_markdown e pulizia
            if not full_text or len(full_text.strip()) < 10:
                try:
                    md_text = doc.export_to_markdown()
                    # Rimuovi formattazione markdown
                    import re
                    full_text = re.sub(r'^#+\s*', '', md_text, flags=re.MULTILINE)  # Rimuovi headers
                    full_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', full_text)  # Rimuovi bold
                    full_text = re.sub(r'\*([^*]+)\*', r'\1', full_text)  # Rimuovi italic
                    full_text = re.sub(r'`([^`]+)`', r'\1', full_text)  # Rimuovi code
                    full_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', full_text)  # Rimuovi links
                    print(f"[Docling] Used export_to_markdown (cleaned): {len(full_text)} chars")
                except Exception as e:
                    print(f"[Docling] export_to_markdown failed: {e}")

            print(f"[Docling] Final extracted text: {len(full_text)} chars")

        if not full_text or len(full_text.strip()) < 10:
            print(f"[Docling] WARNING: Very little text extracted from {file_path}")
            return []

        # Applica chunking token-based con overlap
        chunks = chunk_text_token_based(
            full_text,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens
        )

        print(f"[Docling] Generated {len(chunks)} chunks with token-based chunking")

        # Converti in formato blocchi
        blocks = []
        for idx, chunk_content in enumerate(chunks):
            token_count = len(tokenizer.encode(chunk_content, add_special_tokens=False))
            print(f"[Docling] Chunk {idx + 1}: {token_count} tokens, {len(chunk_content)} chars")

            blocks.append({
                "page": 0,  # Docling non preserva sempre le pagine
                "kind": "docling_chunk",
                "content": chunk_content
            })

        return blocks

    except Exception as e:
        print(f"[Docling] Extraction error for {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.route('/api/upload-progress/<upload_id>', methods=['GET'])
def get_upload_progress(upload_id: str):
    """Ritorna lo stato di progresso per un upload_id"""
    if upload_id not in upload_progress:
        return jsonify({"error": "Upload ID not found"}), 404
    return jsonify(upload_progress[upload_id])


@app.route('/api/create-collection', methods=['POST'])
def create_collection():
    """Crea una nuova collezione in Weaviate"""
    try:
        data = request.get_json(force=True)
        collection_name = data['name']
        config = data['config']

        client = init_weaviate_client(config['weaviateHost'], config['weaviatePort'])

        if collection_name in client.collections.list_all():
            client.close()
            return jsonify({'error': 'Collection already exists'}), 400

        # Determina l'URL corretto per Ollama in base all'ambiente
        # Se Weaviate è in Docker, usa host.docker.internal
        # Altrimenti usa localhost
        ollama_endpoint = config.get('ollamaUrl', 'http://localhost:11434')
        if 'localhost' in ollama_endpoint or '127.0.0.1' in ollama_endpoint:
            # Weaviate è in Docker, quindi deve usare host.docker.internal
            ollama_endpoint = ollama_endpoint.replace('localhost', 'host.docker.internal').replace('127.0.0.1', 'host.docker.internal')

        client.collections.create(
            name=collection_name,
            properties=[
                Property(name="title", data_type=DataType.TEXT, description="Original document name"),
                Property(name="file_id", data_type=DataType.TEXT,
                         description="Internal unique id for this file (per upload)."),
                Property(name="chunk_index", data_type=DataType.INT, description="Chunk index"),
                Property(name="doc_type", data_type=DataType.TEXT, description="Document type"),
                Property(name="content", data_type=DataType.TEXT, description="Document content (chunk)"),
                Property(name="lang", data_type=DataType.TEXT, description="Detected language"),
                Property(name="page", data_type=DataType.INT, description="Document page number"),
                Property(name="block_kind", data_type=DataType.TEXT, description="Chunk type (docling_chunk)"),
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_ollama(
                # api_endpoint=ollama_endpoint,
                model=config.get('embedModel', 'qwen3-embedding:4b')
            ),
        )

        client.close()
        return jsonify({'success': True, 'collection': collection_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete-collection', methods=['POST'])
def delete_collection():
    """Elimina una collezione Weaviate (schema + dati)."""
    try:
        data = request.get_json(force=True)
        collection_name = data["name"]

        client = init_weaviate_client("127.0.0.1", "8080")

        existing = client.collections.list_all()
        if collection_name not in existing:
            client.close()
            return jsonify({"error": "Collection not found"}), 404

        client.collections.delete(collection_name)

        client.close()
        return jsonify({"success": True, "collection": collection_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload-documents', methods=['POST'])
def upload_documents():
    """Carica documenti nella collezione usando Docling per estrazione e chunking."""
    try:
        upload_id = request.form.get("uploadId", None)
        collection_name = request.form['collection']
        config_str = request.form['config']

        try:
            config = json.loads(config_str)
        except Exception:
            config = eval(config_str)

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'Nessun file caricato'}), 400

        total_files = len(files)

        # Progress init
        if upload_id:
            upload_progress[upload_id] = {
                "stage": "starting",
                "current": 0,
                "total": total_files,
                "percent": 0,
                "currentFile": "",
                "cancelled": False  # Flag per cancellazione
            }

            UPLOAD_SESSIONS[upload_id] = {
                "collection": collection_name,
                "weaviateHost": config['weaviateHost'],
                "weaviatePort": config['weaviatePort'],
                "file_ids": [],
            }


        # Tokenizer coerente con modello embedding
        embed_model = config["embedModel"]
        tokenizer = get_tokenizer(embed_model)

        client = init_weaviate_client(config['weaviateHost'], config['weaviatePort'])
        collection = client.collections.get(collection_name)

        print(f"[Upload] Target: {config.get('weaviateHost')}:{config.get('weaviatePort')}")

        processed_files = 0
        uploaded_files = []
        failed_files = []

        # Formati supportati da Docling
        supported_extensions = [
            '.pdf', '.docx', '.pptx', '.html', '.md', '.txt',
            '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'
        ]

        for file_idx, file in enumerate(files):
            # CONTROLLA SE UPLOAD È STATO CANCELLATO
            if upload_id and upload_id in upload_progress:
                if upload_progress[upload_id].get("cancelled", False):
                    print(f"[Upload] Upload {upload_id} cancellato dall'utente, interrompo...")
                    break

            file_id = str(uuid.uuid4())
            title = Path(file.filename).stem

            # Progress: file corrente
            if upload_id:
                upload_progress[upload_id] = {
                    "stage": "processing",
                    "current": file_idx,
                    "total": total_files,
                    "percent": int((file_idx / total_files) * 100),
                    "currentFile": file.filename,
                    "cancelled": False
                }

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            try:
                # Verifica formato supportato
                if not any(file.filename.lower().endswith(ext) for ext in supported_extensions):
                    failed_files.append({
                        "filename": file.filename,
                        "error": "Formato non supportato (usa PDF, DOCX, PPTX, HTML, TXT o immagini)",
                    })
                    continue

                doc_type = Path(file.filename).suffix[1:]  # es. "pdf", "docx"

                # Ottieni preset chunking (default: balanced)
                preset_name = config.get('ingestPreset', 'balanced')
                preset = INGEST_PRESETS.get(preset_name, INGEST_PRESETS['balanced'])

                print(f"[Upload] Using chunking preset: {preset_name} (max_tokens={preset['max_tokens']}, overlap={preset['overlap_tokens']})")

                # Estrazione con Docling + chunking intelligente con overlap
                blocks = extract_with_docling(
                    tmp_path,
                    tokenizer=tokenizer,
                    max_tokens=preset['max_tokens'],
                    overlap_tokens=preset['overlap_tokens']
                )

                if not blocks:
                    failed_files.append({
                        "filename": file.filename,
                        "error": "Nessun contenuto estratto (file vuoto o corrotto?)",
                    })
                    continue

                # DIAGNOSTICA: Statistiche sui chunk generati
                total_chunks = len(blocks)
                chunk_sizes = [len(b.get("content", "")) for b in blocks]
                avg_chunk_size = sum(chunk_sizes) / total_chunks if total_chunks > 0 else 0
                min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
                max_chunk_size = max(chunk_sizes) if chunk_sizes else 0

                print(f"[Upload] CHUNK STATS - File: {file.filename}")
                print(f"  Total chunks: {total_chunks}")
                print(f"  Avg size: {avg_chunk_size:.0f} chars")
                print(f"  Min size: {min_chunk_size} chars")
                print(f"  Max size: {max_chunk_size} chars")
                if total_chunks > 0:
                    sample_chunk = blocks[0].get("content", "")[:200]
                    print(f"  Sample chunk (first 200 chars): {sample_chunk}...")

                # Langdetect su campione globale
                try:
                    sample = ""
                    for b in blocks:
                        c = (b.get("content") or "").strip()
                        if c:
                            sample += c + "\n"
                        if len(sample) >= 2000:
                            sample = sample[:2000]
                            break
                    sample = sample.strip()
                    detected_lang = detect(sample) if sample else "unknown"
                except Exception:
                    detected_lang = "unknown"

                print(f"[Upload] File: {file.filename} - Chunks: {len(blocks)} - Lang: {detected_lang}")

                # AGGIUNGI file_id SUBITO alla sessione (prima di iniziare i chunk)
                # così è disponibile per la cancellazione
                if upload_id and upload_id in UPLOAD_SESSIONS:
                    UPLOAD_SESSIONS[upload_id]["file_ids"].append(file_id)

                total_chunks = len(blocks)

                for idx, block in enumerate(blocks):
                    # CONTROLLA SE UPLOAD È STATO CANCELLATO (anche durante i chunk)
                    if upload_id and upload_id in upload_progress:
                        if upload_progress[upload_id].get("cancelled", False):
                            print(f"[Upload] Upload {upload_id} cancellato durante chunk {idx+1}/{total_chunks}")
                            # Rollback dei chunk di questo file già caricati
                            try:
                                collection.data.delete_many(
                                    where=Filter.by_property("file_id").equal(file_id)
                                )
                                print(f"[Upload] Rimossi chunk parziali del file {file.filename}")
                            except Exception as e:
                                print(f"[Upload] Errore rimozione chunk parziali: {e}")
                            raise Exception("Upload cancellato dall'utente")

                    # Progress: anche durante i chunk
                    if upload_id:
                        file_progress = (idx / total_chunks) if total_chunks > 0 else 0
                        overall_progress = (file_idx + file_progress) / total_files
                        upload_progress[upload_id] = {
                            "stage": "processing",
                            "current": file_idx,
                            "total": total_files,
                            "percent": int(overall_progress * 100),
                            "currentFile": f"{file.filename} (chunk {idx + 1}/{total_chunks})",
                            "cancelled": False
                        }

                    chunk_text = block["content"]

                    # CONTROLLA CANCELLAZIONE PRIMA DELL'EMBEDDING (operazione lenta)
                    if upload_id and upload_id in upload_progress:
                        if upload_progress[upload_id].get("cancelled", False):
                            print(f"[Upload] Cancellato prima di embedding chunk {idx+1}/{total_chunks}")
                            raise Exception("Upload cancellato dall'utente")

                    embedding = get_ollama_embedding(
                        chunk_text,
                        config['ollamaUrl'],
                        config['embedModel']
                    )

                    # CONTROLLA CANCELLAZIONE ANCHE DOPO EMBEDDING (prima dell'insert)
                    if upload_id and upload_id in upload_progress:
                        if upload_progress[upload_id].get("cancelled", False):
                            print(f"[Upload] Cancellato dopo embedding chunk {idx+1}/{total_chunks}")
                            raise Exception("Upload cancellato dall'utente")

                    result = collection.data.insert(
                        properties={
                            "title": title,
                            "file_id": file_id,
                            "chunk_index": idx,
                            "doc_type": doc_type,
                            "content": chunk_text,
                            "lang": detected_lang,
                            "page": block.get("page", 0),
                            "block_kind": block.get("kind", "docling_chunk"),
                        },
                        vector=embedding,
                    )

                    print(f"[Upload] Inserted chunk {idx + 1}/{total_chunks} - UUID: {result}")

                processed_files += 1
                uploaded_files.append(file.filename)
                # file_id già aggiunto all'inizio, non serve duplicare

            except Exception as file_error:
                # Controlla se è una cancellazione
                if "Upload cancellato" in str(file_error):
                    print(f"[Upload] Cancellazione rilevata, interrompo loop...")
                    break  # Esce dal loop FOR

                # rollback file in caso di errore
                try:
                    collection.data.delete_many(
                        where=Filter.by_property("file_id").equal(file_id)
                    )
                except Exception:
                    pass

                raw_msg = str(file_error)
                if "surrogates not allowed" in raw_msg:
                    friendly_msg = (
                        "Il file contiene caratteri speciali non supportati "
                        "(es. simboli/font particolari). Prova a riesportarlo come PDF standard."
                    )
                else:
                    friendly_msg = raw_msg

                failed_files.append({
                    "filename": file.filename,
                    "error": friendly_msg,
                })

            finally:
                os.unlink(tmp_path)

        # Progress done (o cancelled se interrotto)
        if upload_id:
            # Controlla se è stato cancellato
            was_cancelled = upload_id in upload_progress and upload_progress[upload_id].get("cancelled", False)

            if was_cancelled:
                upload_progress[upload_id] = {
                    "stage": "cancelled",
                    "current": file_idx,
                    "total": total_files,
                    "percent": int((file_idx / total_files) * 100) if total_files > 0 else 0,
                    "currentFile": "",
                    "cancelled": True
                }
                print(f"[Upload] Upload {upload_id} completamente cancellato")
            else:
                upload_progress[upload_id] = {
                    "stage": "done",
                    "current": total_files,
                    "total": total_files,
                    "percent": 100,
                    "currentFile": "",
                    "cancelled": False
                }

        client.close()

        return jsonify({
            "success": len(failed_files) == 0,
            "processedFiles": processed_files,
            "uploadedFiles": uploaded_files,
            "failedFiles": failed_files,
            "collection": collection_name,
        })

    except Exception as e:
        # Progress error
        if upload_id:
            upload_progress[upload_id] = {
                "stage": "error",
                "current": 0,
                "total": 0,
                "percent": 0,
                "error": str(e)
            }
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route("/api/cancel-upload", methods=["POST"])
def cancel_upload():
    """
    Annulla un upload in corso o appena terminato e rimuove dal DB
    i documenti già indicizzati per quell'uploadId.
    """
    try:
        data = request.get_json(force=True)
        upload_id = data.get("uploadId")

        if not upload_id:
            return jsonify({"error": "uploadId mancante"}), 400

        # IMPOSTA FLAG CANCELLED PER INTERROMPERE IL LOOP
        if upload_id in upload_progress:
            upload_progress[upload_id]["cancelled"] = True
            upload_progress[upload_id]["stage"] = "cancelling"
            print(f"[Cancel] Impostato flag cancelled per upload {upload_id}")

        session = UPLOAD_SESSIONS.get(upload_id)
        if not session:
            # Se non c'è sessione, elimina solo il progress
            upload_progress.pop(upload_id, None)
            return jsonify({"success": True, "deleted": 0})

        collection_name = session.get("collection")
        host = session.get("weaviateHost", "127.0.0.1")
        port = session.get("weaviatePort", "8080")
        file_ids = session.get("file_ids", [])

        if not collection_name or not file_ids:
            UPLOAD_SESSIONS.pop(upload_id, None)
            upload_progress.pop(upload_id, None)
            return jsonify({"success": True, "deleted": 0})

        # ELIMINA TUTTI I CHUNK GIÀ CARICATI
        client = init_weaviate_client(host, port)
        collection = client.collections.get(collection_name)

        deleted = 0
        for fid in file_ids:
            try:
                result = collection.data.delete_many(
                    where=Filter.by_property("file_id").equal(fid)
                )
                deleted += 1
                print(f"[Cancel] Eliminati chunk del file {fid}")
            except Exception as e:
                print(f"[Cancel] Errore eliminazione file {fid}: {e}")

        client.close()

        # PULISCI TUTTO
        upload_progress.pop(upload_id, None)
        UPLOAD_SESSIONS.pop(upload_id, None)

        print(f"[Cancel] Upload {upload_id} cancellato, eliminati {deleted} file")
        return jsonify({"success": True, "deleted": deleted})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/collections', methods=['GET'])
def get_collections():
    """Ottieni lista collezioni"""
    try:
        host = request.args.get('host', '127.0.0.1')
        port = request.args.get('port', '8080')

        client = init_weaviate_client(host, port)
        collections = client.collections.list_all()
        client.close()

        return jsonify({'collections': [{'name': col} for col in collections]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'ok'})


@app.route("/api/debug-count/<collection_name>", methods=["GET"])
def debug_count(collection_name):
    host = request.args.get("host", "127.0.0.1")
    port = request.args.get("port", "8080")
    client = init_weaviate_client(host, port)
    col = client.collections.get(collection_name)

    print(f"[Debug] Counting collection: {collection_name}")

    res = col.aggregate.over_all(total_count=True)
    client.close()
    return jsonify({"collection": collection_name, "total_count": res.total_count})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)