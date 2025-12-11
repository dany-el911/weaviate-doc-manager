from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from pathlib import Path
import requests
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import Filter
from pypdf import PdfReader
from typing import Dict, Any
import tempfile
import os
import uuid
import json
import time

app = Flask(__name__)
CORS(app)

# Dizionario globale per tracciare il progresso degli upload
upload_progress = {}

# Traccia gli upload per poterli cancellare:
# upload_id -> { collection, weaviateHost, weaviatePort, file_ids }
UPLOAD_SESSIONS: Dict[str, Dict[str, Any]] = {}

def normalize_text(text: str) -> str:
    """
    Rimuove caratteri Unicode non validi per UTF-8 (es. surrogates),
    in modo da evitare errori 'surrogates not allowed'.
    """
    return text.encode("utf-8", "ignore").decode("utf-8")


def get_ollama_embedding(text: str, ollama_url: str, model: str) -> list[float]:
    """Genera embedding tramite Ollama"""
    max_length = 2048
    if len(text) > max_length:
        text = text[:max_length]

    payload = {
        "model": model,
        "prompt": text,
    }
    resp = requests.post(f"{ollama_url}/api/embeddings", json=payload, timeout=120)
    if not resp.ok:
        raise Exception(f"Ollama error: {resp.status_code}")
    data = resp.json()
    return data["embedding"]


def extract_text_from_pdf(file_path: str) -> str:
    """Estrae testo da PDF"""
    reader = PdfReader(file_path)
    texts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            texts.append(page_text.strip())
    return "\n\n".join(texts).strip()


def chunk_text(text: str, chunk_size: int) -> list[str]:
    """
    Suddivide il testo in chunk con overlap fisso = 20%.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    overlap = int(chunk_size * 0.20)
    stride = chunk_size - overlap

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += stride

    return chunks


def init_weaviate_client(host: str, port: str):
    """Inizializza client Weaviate"""
    return weaviate.connect_to_local(
        host=host,
        port=int(port),
        grpc_port=50051,
    )


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
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="Original document name",
                ),
                Property(
                    name="file_id",
                    data_type=DataType.TEXT,
                    description="Internal unique id for this file (per upload).",
                ),
                Property(
                    name="chunk_index",
                    data_type=DataType.INT,
                    description="Chunk index",
                ),
                Property(
                    name="doc_type",
                    data_type=DataType.TEXT,
                    description="Document type",
                ),
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="Document content",
                ),
            ],
            vectorizer_config=Configure.Vectorizer.none(),
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
    """Carica documenti nella collezione, con progress tracking."""
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

        # ✅ Inizializza il progresso per questo uploadId
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

        client = init_weaviate_client(config['weaviateHost'], config['weaviatePort'])
        collection = client.collections.get(collection_name)

        processed_files = 0
        uploaded_files = []
        failed_files = []

        try:
            chunk_size = int(config.get('chunkSize', 2000))
        except Exception:
            chunk_size = 2000

        for file_idx, file in enumerate(files):
            file_id = str(uuid.uuid4())
            title = Path(file.filename).stem

            # ✅ Aggiorna il progresso: file corrente
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
                if file.filename.endswith('.pdf'):
                    text = extract_text_from_pdf(tmp_path)
                    text = normalize_text(text)
                    doc_type = "pdf"
                elif file.filename.endswith('.txt'):
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    text = normalize_text(text)
                    doc_type = "txt"
                else:
                    failed_files.append({
                        "filename": file.filename,
                        "error": "Formato non supportato (usa PDF o TXT)",
                    })
                    continue

                if not text:
                    failed_files.append({
                        "filename": file.filename,
                        "error": "Nessun testo leggibile trovato",
                    })
                    continue

                chunks = chunk_text(text, chunk_size)
                total_chunks = len(chunks)

                for idx, chunk in enumerate(chunks):
                    # ✅ Aggiorna progresso anche durante i chunk
                    if upload_id:
                        # Progresso del file corrente basato sui chunk
                        file_progress = (idx / total_chunks) if total_chunks > 0 else 0
                        overall_progress = (file_idx + file_progress) / total_files

                        upload_progress[upload_id] = {
                            "stage": "processing",
                            "current": file_idx,
                            "total": total_files,
                            "percent": int(overall_progress * 100),
                            "currentFile": f"{file.filename} (chunk {idx + 1}/{total_chunks})"
                        }

                    embedding = get_ollama_embedding(
                        chunk,
                        config['ollamaUrl'],
                        config['embedModel']
                    )

                    collection.data.insert(
                        properties={
                            "title": title,
                            "file_id": file_id,
                            "chunk_index": idx,
                            "doc_type": doc_type,
                            "content": chunk,
                        },
                        vector=embedding,
                    )

                processed_files += 1
                uploaded_files.append(file.filename)

                if upload_id and upload_id in UPLOAD_SESSIONS:
                    UPLOAD_SESSIONS[upload_id]["file_ids"].append(file_id)

            except Exception as file_error:
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
                        "(ad esempio simboli matematici o font particolari). "
                        "Prova a riesportarlo come PDF standard o testo semplice."
                    )
                else:
                    friendly_msg = raw_msg

                failed_files.append({
                    "filename": file.filename,
                    "error": friendly_msg,
                })

            finally:
                os.unlink(tmp_path)

        # ✅ Segna come completato
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
        # ✅ Segna come errore nel progresso
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
            # Nessuna info per questo uploadId: niente da cancellare
            return jsonify({"error": "Upload non trovato"}), 404

        collection_name = session.get("collection")
        host = session.get("weaviateHost", "127.0.0.1")
        port = session.get("weaviatePort", "8080")
        file_ids = session.get("file_ids", [])

        # Se non ci sono file indicizzati, non c'è nulla da cancellare
        if not collection_name or not file_ids:
            # rimuoviamo comunque la sessione per pulizia
            UPLOAD_SESSIONS.pop(upload_id, None)
            return jsonify({"success": True, "deleted": 0})

        # Connessione a Weaviate
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
                # se la cancellazione di un file fallisce, continuiamo con gli altri
                pass

        client.close()

        # Rimuovi la sessione perché è stata annullata/chiusa
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)