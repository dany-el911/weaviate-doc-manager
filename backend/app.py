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

# PDF digitali (testo + tabelle) + OCR fallback
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

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
# CONFIG / PRESETS
# -----------------------------
INGEST_PRESETS = {
    "precision":    {"max_tokens": 384,  "overlap_tokens": 64},
    "balanced":     {"max_tokens": 640,  "overlap_tokens": 80},
    "long_context": {"max_tokens": 1024, "overlap_tokens": 96},
}

# OCR multilingua - lingue più comuni
# Tesseract userà tutti i language pack disponibili in ordine di priorità
OCR_LANGS = "ita+eng+deu+fra+spa+por+nld+pol+rus"

_TOKENIZER_CACHE: Dict[str, Any] = {}


# -----------------------------
# Helpers
# -----------------------------
def get_available_ocr_langs() -> str:
    """
    Rileva automaticamente le lingue OCR disponibili in Tesseract.
    Se alcune lingue non sono installate, usa solo quelle disponibili.
    Fallback a 'eng' se nessuna delle lingue configurate è disponibile.
    """
    try:
        available = pytesseract.get_languages()
        # Lista di priorità: italiano, inglese, tedesco, francese, spagnolo, portoghese, olandese, polacco, russo
        preferred = ['ita', 'eng', 'deu', 'fra', 'spa', 'por', 'nld', 'pol', 'rus']

        # Filtra solo le lingue installate
        found = [lang for lang in preferred if lang in available]

        if found:
            result = '+'.join(found)
            print(f"[OCR] Lingue disponibili: {result}")
            return result
        else:
            print("[OCR] WARNING: Nessuna lingua preferita trovata, uso 'eng' di default")
            return 'eng'
    except Exception as e:
        print(f"[OCR] Errore rilevamento lingue: {e}, uso configurazione di default")
        return OCR_LANGS


# Rileva le lingue OCR all'avvio
DETECTED_OCR_LANGS = get_available_ocr_langs()


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
    Se il repo HF non coincide nel tuo setup, cambia qui.
    """
    m = (embed_model or "").lower()

    if "qwen3-embedding" in m:
        # Repo HF tipico; se non esiste nel tuo ambiente, sostituiscilo col nome corretto
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
    IMPORTANTE: niente truncate a caratteri qui.
    La dimensione è garantita dal chunking token-based.
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


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_rtf(file_path: str) -> str:
    """
    Estrae testo da file RTF (Rich Text Format).
    Rimuove la formattazione RTF e restituisce solo il testo.
    """
    try:
        from striprtf.striprtf import rtf_to_text
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            rtf_content = f.read()
        text = rtf_to_text(rtf_content)
        return text or ""
    except Exception as e:
        print(f"RTF extraction error for {file_path}: {e}")
        # Fallback: prova a leggere come testo semplice (toglie alcuni tag RTF base)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # Rimuovi tag RTF basilari
            import re
            text = re.sub(r'\\[a-z]+\d*\s?', ' ', content)  # Rimuovi comandi RTF
            text = re.sub(r'[{}]', '', text)  # Rimuovi parentesi graffe
            text = re.sub(r'\s+', ' ', text)  # Normalizza spazi
            return text.strip()
        except Exception as e2:
            print(f"RTF fallback error for {file_path}: {e2}")
            return ""


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocessa un'immagine per migliorare la qualità dell'OCR:
    - Conversione in scala di grigi
    - Aumento del contrasto
    - Sharpening
    """
    # Converti in scala di grigi
    if image.mode != 'L':
        image = image.convert('L')

    # Aumenta il contrasto
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Applica sharpening
    image = image.filter(ImageFilter.SHARPEN)

    return image


def extract_text_from_image(file_path: str) -> str:
    """
    Estrae testo da file immagine usando OCR con preprocessing.
    Supporta: JPG, JPEG, PNG, TIFF, BMP
    Usa automaticamente tutte le lingue disponibili in Tesseract.
    """
    try:
        image = Image.open(file_path)
        # Preprocessa per migliorare OCR
        processed_image = preprocess_image_for_ocr(image)
        # Esegui OCR con lingue rilevate dinamicamente
        text = pytesseract.image_to_string(processed_image, lang=DETECTED_OCR_LANGS) or ""
        return normalize_text(text).strip()
    except Exception as e:
        print(f"OCR ERROR on image {file_path}: {e}")
        return ""


def extract_pdf_blocks(file_path: str) -> List[Dict[str, Any]]:
    """
    Estrae blocchi da PDF:
      - testo digitale per pagina (kind='text')
      - tabelle (kind='table')
      - OCR fallback per pagina se testo quasi vuoto (kind='ocr_text')

    Ritorna una lista di dict: {page, kind, content}
    """
    blocks: List[Dict[str, Any]] = []

    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            # testo digitale
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            page_text = normalize_text(page_text).strip()

            # tabelle digitali
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            # OCR se pagina “vuota” e niente tabelle
            needs_ocr = (len(page_text) < 30) and (len(tables) == 0)

            if needs_ocr:
                ocr_text = ""
                try:
                    images = convert_from_path(
                        file_path,
                        first_page=page_idx,
                        last_page=page_idx,
                        dpi=200,
                    )
                    if images:
                        # Usa preprocessing per migliorare OCR
                        processed_image = preprocess_image_for_ocr(images[0])
                        # Usa lingue rilevate dinamicamente
                        ocr_text = pytesseract.image_to_string(processed_image, lang=DETECTED_OCR_LANGS) or ""
                        ocr_text = normalize_text(ocr_text).strip()
                except Exception as e:
                    print(f"OCR ERROR on PDF page {page_idx}: {e}")
                    ocr_text = ""

                if ocr_text:
                    blocks.append({"page": page_idx, "kind": "ocr_text", "content": ocr_text})
                continue

            if page_text:
                blocks.append({"page": page_idx, "kind": "text", "content": page_text})

            # linearizza tabelle
            for t in tables:
                rows = []
                for row in t:
                    if not row:
                        continue
                    clean = [("" if c is None else str(c).strip()) for c in row]
                    line = " | ".join(clean).strip()
                    if line:
                        rows.append(line)

                table_text = "\n".join(rows).strip()
                if table_text:
                    blocks.append({"page": page_idx, "kind": "table", "content": table_text})

    return blocks


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

        client.collections.create(
            name=collection_name,
            properties=[
                Property(name="title", data_type=DataType.TEXT, description="Original document name"),
                Property(name="file_id", data_type=DataType.TEXT, description="Internal unique id for this file (per upload)."),
                Property(name="chunk_index", data_type=DataType.INT, description="Chunk index"),
                Property(name="doc_type", data_type=DataType.TEXT, description="Document type"),
                Property(name="content", data_type=DataType.TEXT, description="Document content (chunk)"),

                # nuovi metadati utili
                Property(name="lang", data_type=DataType.TEXT, description="Detected language"),
                Property(name="page", data_type=DataType.INT, description="PDF page number (1-based)"),
                Property(name="block_kind", data_type=DataType.TEXT, description="text | table | ocr_text"),
                Property(name="ingest_mode", data_type=DataType.TEXT, description="precision | balanced | long_context"),
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_ollama(
                api_endpoint="http://host.docker.internal:11434",
                model="qwen3-embedding:4b"
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
    """Carica documenti nella collezione, con progress tracking + token chunking + OCR automatico."""
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
                "currentFile": ""
            }

            UPLOAD_SESSIONS[upload_id] = {
                "collection": collection_name,
                "weaviateHost": config['weaviateHost'],
                "weaviatePort": config['weaviatePort'],
                "file_ids": [],
            }

        # Ingest mode presets
        ingest_mode = (config.get("ingestMode") or "balanced").strip()
        preset = INGEST_PRESETS.get(ingest_mode, INGEST_PRESETS["balanced"])
        max_tokens = preset["max_tokens"]
        overlap_tokens = preset["overlap_tokens"]

        # Tokenizer coerente con modello embedding
        embed_model = config["embedModel"]
        tokenizer = get_tokenizer(embed_model)

        client = init_weaviate_client(config['weaviateHost'], config['weaviatePort'])
        collection = client.collections.get(collection_name)

        print("WEAVIATE TARGET UPLOAD:", config.get("weaviateHost"), config.get("weaviatePort"))

        processed_files = 0
        uploaded_files = []
        failed_files = []

        for file_idx, file in enumerate(files):
            file_id = str(uuid.uuid4())
            title = Path(file.filename).stem

            # Progress: file corrente
            if upload_id:
                upload_progress[upload_id] = {
                    "stage": "processing",
                    "current": file_idx,
                    "total": total_files,
                    "percent": int((file_idx / total_files) * 100),
                    "currentFile": file.filename
                }

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            try:
                doc_type = None
                blocks: List[Dict[str, Any]] = []

                if file.filename.endswith('.pdf'):
                    doc_type = "pdf"
                    blocks = extract_pdf_blocks(tmp_path)
                elif file.filename.endswith('.txt'):
                    doc_type = "txt"
                    txt = extract_text_from_txt(tmp_path)
                    txt = normalize_text(txt).strip()
                    if txt:
                        blocks = [{"page": 0, "kind": "text", "content": txt}]
                elif file.filename.lower().endswith('.rtf'):
                    doc_type = "rtf"
                    txt = extract_text_from_rtf(tmp_path)
                    txt = normalize_text(txt).strip()
                    if txt:
                        blocks = [{"page": 0, "kind": "text", "content": txt}]
                elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp')):
                    doc_type = "image"
                    txt = extract_text_from_image(tmp_path)
                    if txt:
                        blocks = [{"page": 0, "kind": "ocr_text", "content": txt}]
                else:
                    failed_files.append({
                        "filename": file.filename,
                        "error": "Formato non supportato (usa PDF, TXT, RTF o immagini JPG/PNG/TIFF/BMP)",
                    })
                    continue

                if not blocks:
                    failed_files.append({
                        "filename": file.filename,
                        "error": "Nessun testo leggibile trovato (PDF vuoto / OCR fallito?)",
                    })
                    continue

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

                # Chunking per blocco (manteniamo page/kind)
                all_chunks: List[Dict[str, Any]] = []
                for b in blocks:
                    content = normalize_text(b.get("content", "")).strip()
                    if not content:
                        continue

                    # Tabelle: niente overlap (più pulito)
                    local_overlap = 0 if b.get("kind") == "table" else overlap_tokens

                    chunks = chunk_text_token_based(
                        content,
                        tokenizer=tokenizer,
                        max_tokens=max_tokens,
                        overlap_tokens=local_overlap,
                    )

                    for c in chunks:
                        all_chunks.append({
                            "page": int(b.get("page", 0)),
                            "block_kind": b.get("kind", "text"),
                            "content": c,
                        })

                # DEBUG: dopo all_chunks
                print("COLLECTION:", collection_name)
                print("FILES:", len(files))
                print("INGEST MODE:", ingest_mode, "max_tokens:", max_tokens, "overlap:", overlap_tokens)
                print(f"FILE: {file.filename} - CHUNKS: {len(all_chunks)}")

                if not all_chunks:
                    failed_files.append({
                        "filename": file.filename,
                        "error": "Contenuto presente ma chunking ha prodotto 0 chunk",
                    })
                    continue

                total_chunks = len(all_chunks)

                for idx, ch in enumerate(all_chunks):
                    # Progress: anche durante i chunk
                    if upload_id:
                        file_progress = (idx / total_chunks) if total_chunks > 0 else 0
                        overall_progress = (file_idx + file_progress) / total_files
                        upload_progress[upload_id] = {
                            "stage": "processing",
                            "current": file_idx,
                            "total": total_files,
                            "percent": int(overall_progress * 100),
                            "currentFile": f"{file.filename} (chunk {idx + 1}/{total_chunks})"
                        }

                    chunk_text_value = ch["content"]

                    # DEBUG: durante insert
                    print(f"INSERT chunk {idx + 1}/{total_chunks} - page: {ch.get('page', 0)} - kind: {ch.get('block_kind', 'text')}")

                    embedding = get_ollama_embedding(
                        chunk_text_value,
                        config['ollamaUrl'],
                        config['embedModel']
                    )

                    # DEBUG: verifica embedding
                    print(f"EMBEDDING generato: len={len(embedding)}")

                    result = collection.data.insert(
                        properties={
                            "title": title,
                            "file_id": file_id,
                            "chunk_index": idx,
                            "doc_type": doc_type,
                            "content": chunk_text_value,
                            "lang": detected_lang,
                            "page": ch.get("page", 0),
                            "block_kind": ch.get("block_kind", "text"),
                            "ingest_mode": ingest_mode,
                        },
                        vector=embedding,
                    )

                    # DEBUG: conferma insert
                    print(f"INSERT OK chunk {idx + 1} - UUID: {result}")

                processed_files += 1
                uploaded_files.append(file.filename)

                if upload_id and upload_id in UPLOAD_SESSIONS:
                    UPLOAD_SESSIONS[upload_id]["file_ids"].append(file_id)

            except Exception as file_error:
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

        # Progress done
        if upload_id:
            upload_progress[upload_id] = {
                "stage": "done",
                "current": total_files,
                "total": total_files,
                "percent": 100,
                "currentFile": ""
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

        session = UPLOAD_SESSIONS.get(upload_id)
        if not session:
            return jsonify({"error": "Upload non trovato"}), 404

        collection_name = session.get("collection")
        host = session.get("weaviateHost", "127.0.0.1")
        port = session.get("weaviatePort", "8080")
        file_ids = session.get("file_ids", [])

        if not collection_name or not file_ids:
            UPLOAD_SESSIONS.pop(upload_id, None)
            return jsonify({"success": True, "deleted": 0})

        client = init_weaviate_client(host, port)
        collection = client.collections.get(collection_name)

        deleted = 0
        for fid in file_ids:
            try:
                collection.data.delete_many(
                    where=Filter.by_property("file_id").equal(fid)
                )
                deleted += 1
            except Exception:
                pass

        client.close()

        upload_progress.pop(upload_id, None)
        UPLOAD_SESSIONS.pop(upload_id, None)

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

    print("WEAVIATE TARGET COUNT:", host, port)

    res = col.aggregate.over_all(total_count=True)
    client.close()
    return jsonify({"collection": collection_name, "total_count": res.total_count})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
