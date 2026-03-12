import collections
import datetime
import json
import os
import platform
import re
import tempfile
import time
import traceback
import uuid
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse, urlunparse

from flask import Flask, jsonify, request
from flask_cors import CORS
from langdetect import detect
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoTokenizer

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

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
    "precision": {"max_tokens": 384, "overlap_tokens": 64},
    "balanced": {"max_tokens": 1024, "overlap_tokens": 128},
    "long_context": {"max_tokens": 2048, "overlap_tokens": 192},
}

# Stopwords inglesi da rimuovere per disattivare lo stopwording
ENGLISH_STOPWORDS = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"
]

_TOKENIZER_CACHE: Dict[str, Any] = {}
_OLLAMA_ENDPOINT_CACHE: Dict[str, str] = {}

# Profili delle collection, aggiornati dopo ogni ingestione
# { collection_name -> dict }
COLLECTION_PROFILES: Dict[str, Dict[str, Any]] = {}

# Stopwords generiche multilingua usate per estrarre top_terms
_PROFILE_STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "una", "uno",
    "e", "o", "ma", "se", "di", "da", "in", "con", "su", "per",
    "tra", "fra", "a", "al", "del", "della", "dei", "degli", "delle",
    "che", "è", "non", "si", "ha", "ho", "ci", "ne", "mi", "ti",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "of", "for", "is", "it", "as", "by", "be", "this", "that", "with",
    "are", "was", "not", "from", "they", "we", "you", "he", "she",
    "der", "die", "das", "und", "in", "zu", "den", "des", "dem", "ein",
    "le", "la", "les", "de", "du", "des", "et", "en", "un", "une",
    "que", "se", "por", "con", "del", "los", "las", "una", "al",
}


class CollectionProfileBuilder:
    """
    Accumula metadati aggregati sui documenti/chunk ingeriti in una collection.
    Va istanziato una volta per sessione di ingestione e aggiornato
    con ogni file processato. Il metodo build() restituisce il profilo finale.

    Il campo _term_counts viene mantenuto come dict serializzabile in modo da
    poter essere ripristinato e riaggiornato nelle ingestioni successive,
    evitando il degrado del merge ingenuo su liste finali.
    """

    MAX_SAMPLE_TITLES = 5
    MAX_SAMPLE_CHUNKS = 5
    MAX_SAMPLE_CHUNK_CHARS = 250  # lunghezza massima per singolo sample chunk
    TOP_TERMS_N = 20

    def __init__(
        self,
        collection_name: str,
        searchable_fields: List[str],
        content_fields: List[str],
        *,
        seed_term_counts: Dict[str, int] | None = None,
    ):
        self.collection_name = collection_name
        self.searchable_fields = searchable_fields
        self.content_fields = content_fields

        self._document_count: int = 0
        self._chunk_count: int = 0
        self._languages: collections.Counter = collections.Counter()
        self._doc_types: collections.Counter = collections.Counter()
        self._sample_titles: List[str] = []
        self._sample_chunks: List[str] = []
        # _term_counts è il contatore grezzo persistibile (dict serializzabile)
        self._term_counts: collections.Counter = collections.Counter(seed_term_counts or {})

    def add_document(
        self,
        title: str,
        doc_type: str,
        detected_lang: str,
        chunks: List[str],
    ) -> None:
        """Registra un documento e i suoi chunk nel profilo."""
        self._document_count += 1
        self._chunk_count += len(chunks)
        self._doc_types[doc_type] += 1

        if detected_lang and detected_lang != "unknown":
            self._languages[detected_lang] += 1

        if title and len(self._sample_titles) < self.MAX_SAMPLE_TITLES:
            if title not in self._sample_titles:
                self._sample_titles.append(title)

        for chunk in chunks:
            cleaned = " ".join(chunk.split())  # normalizza spazi e newline
            if cleaned and len(self._sample_chunks) < self.MAX_SAMPLE_CHUNKS:
                self._sample_chunks.append(cleaned[:self.MAX_SAMPLE_CHUNK_CHARS])

            # Accumula termini per top_terms nel contatore grezzo
            self._update_term_counter(chunk)

    def _update_term_counter(self, text: str) -> None:
        """Estrae token normalizzati dal testo e li conta."""
        tokens = re.findall(r"\b[a-zA-ZàáâãäåæçèéêëìíîïðñòóôõöùúûüýÀ-Ö]{3,}\b", text.lower())
        for tok in tokens:
            if tok not in _PROFILE_STOPWORDS:
                self._term_counts[tok] += 1

    @staticmethod
    def build_description_text(
        collection_name: str,
        document_count: int,
        chunk_count: int,
        languages_distribution: Dict[str, int],
        doc_types_distribution: Dict[str, int],
        searchable_fields: List[str],
        top_terms: List[str],
        sample_titles: List[str],
    ) -> str:
        """
        Genera una stringa descrittiva in linguaggio naturale della collection.
        Usata per embedding, BM25, similarity search e reranking.
        """
        langs = ", ".join(
            lang for lang, _ in sorted(languages_distribution.items(), key=lambda x: -x[1])
        ) or "sconosciuta"
        doc_types = ", ".join(
            dt for dt, _ in sorted(doc_types_distribution.items(), key=lambda x: -x[1])
        ) or "vari"
        terms_str = ", ".join(top_terms[:10]) if top_terms else "n/d"
        titles_str = ", ".join(sample_titles[:3]) if sample_titles else "n/d"
        fields_str = ", ".join(searchable_fields)

        return (
            f"Collection {collection_name}. "
            f"{document_count} documenti, {chunk_count} chunk. "
            f"Lingue principali: {langs}. "
            f"Tipi documento: {doc_types}. "
            f"Campi ricercabili: {fields_str}. "
            f"Termini frequenti: {terms_str}. "
            f"Titoli campione: {titles_str}."
        )

    def build(self) -> Dict[str, Any]:
        """Restituisce il profilo aggregato della collection."""
        top_terms = [term for term, _ in self._term_counts.most_common(self.TOP_TERMS_N)]

        langs = dict(self._languages)
        doc_types = dict(self._doc_types)

        description_text = self.build_description_text(
            collection_name=self.collection_name,
            document_count=self._document_count,
            chunk_count=self._chunk_count,
            languages_distribution=langs,
            doc_types_distribution=doc_types,
            searchable_fields=self.searchable_fields,
            top_terms=top_terms,
            sample_titles=self._sample_titles,
        )

        return {
            "collection_name": self.collection_name,
            "description_text": description_text,
            "document_count": self._document_count,
            "chunk_count": self._chunk_count,
            "languages_distribution": langs,
            "doc_types_distribution": doc_types,
            "searchable_fields": self.searchable_fields,
            "content_fields": self.content_fields,
            "sample_titles": self._sample_titles,
            "sample_chunks": self._sample_chunks,
            "top_terms": top_terms,
            # contatore grezzo serializzabile: permette merge corretto nelle ingestioni future
            "_term_counts": dict(self._term_counts),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


# Campi testuali standard della collection (usati nel profilo)
_COLLECTION_SEARCHABLE_FIELDS = ["title", "content", "doc_type", "lang"]
_COLLECTION_CONTENT_FIELDS = ["content"]

# Collection tecnica Weaviate dove vengono persistiti i profili
_PROFILES_COLLECTION_NAME = "ELYSIA_COLLECTION_PROFILES"


def _ensure_profiles_collection(client) -> None:
    """Crea la collection tecnica dei profili se non esiste già."""
    existing = client.collections.list_all()
    if _PROFILES_COLLECTION_NAME in existing:
        return

    client.collections.create(
        name=_PROFILES_COLLECTION_NAME,
        description="Profili descrittivi delle collection del corpus (metadati di sistema).",
        properties=[
            Property(name="collection_name", data_type=DataType.TEXT,
                     description="Nome della collection reale"),
            Property(name="description_text", data_type=DataType.TEXT,
                     description="Testo descrittivo in linguaggio naturale (per embedding/BM25/reranking)"),
            Property(name="document_count", data_type=DataType.INT,
                     description="Numero totale di documenti ingeriti"),
            Property(name="chunk_count", data_type=DataType.INT,
                     description="Numero totale di chunk ingeriti"),
            Property(name="languages_json", data_type=DataType.TEXT,
                     description="Distribuzione delle lingue (JSON)"),
            Property(name="doc_types_json", data_type=DataType.TEXT,
                     description="Distribuzione dei tipi documento (JSON)"),
            Property(name="searchable_fields_json", data_type=DataType.TEXT,
                     description="Campi ricercabili (JSON array)"),
            Property(name="content_fields_json", data_type=DataType.TEXT,
                     description="Campi contenuto (JSON array)"),
            Property(name="sample_titles_text", data_type=DataType.TEXT,
                     description="Titoli campione separati da newline"),
            Property(name="sample_chunks_text", data_type=DataType.TEXT,
                     description="Chunk campione separati da doppio newline"),
            Property(name="top_terms_text", data_type=DataType.TEXT,
                     description="Termini frequenti separati da virgola"),
            Property(name="term_counts_json", data_type=DataType.TEXT,
                     description="Contatore grezzo dei termini (JSON) — usato per merge corretto"),
            Property(name="updated_at", data_type=DataType.TEXT,
                     description="Timestamp ultimo aggiornamento (ISO 8601 UTC)"),
        ],
        # Nessun vectorizer: è una collection di metadati strutturati, non di corpus
        vectorizer_config=Configure.Vectorizer.none(),
    )
    print(f"[Profiles] Collection tecnica '{_PROFILES_COLLECTION_NAME}' creata.")


def _profile_to_weaviate_properties(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Converte il dict profilo nei campi Weaviate (tutto scalare/JSON string)."""
    return {
        "collection_name": profile["collection_name"],
        "description_text": profile.get("description_text", ""),
        "document_count": profile.get("document_count", 0),
        "chunk_count": profile.get("chunk_count", 0),
        "languages_json": json.dumps(profile.get("languages_distribution", {}), ensure_ascii=False),
        "doc_types_json": json.dumps(profile.get("doc_types_distribution", {}), ensure_ascii=False),
        "searchable_fields_json": json.dumps(profile.get("searchable_fields", []), ensure_ascii=False),
        "content_fields_json": json.dumps(profile.get("content_fields", []), ensure_ascii=False),
        "sample_titles_text": "\n".join(profile.get("sample_titles", [])),
        "sample_chunks_text": "\n\n".join(profile.get("sample_chunks", [])),
        "top_terms_text": ", ".join(profile.get("top_terms", [])),
        "term_counts_json": json.dumps(profile.get("_term_counts", {}), ensure_ascii=False),
        "updated_at": profile.get("updated_at", ""),
    }


def _weaviate_properties_to_profile(props: Dict[str, Any]) -> Dict[str, Any]:
    """Riconverte le proprietà Weaviate nel dict profilo normalizzato."""
    def _safe_json(s, default):
        try:
            return json.loads(s) if s else default
        except Exception:
            return default

    return {
        "collection_name": props.get("collection_name", ""),
        "description_text": props.get("description_text", ""),
        "document_count": props.get("document_count", 0),
        "chunk_count": props.get("chunk_count", 0),
        "languages_distribution": _safe_json(props.get("languages_json"), {}),
        "doc_types_distribution": _safe_json(props.get("doc_types_json"), {}),
        "searchable_fields": _safe_json(props.get("searchable_fields_json"), []),
        "content_fields": _safe_json(props.get("content_fields_json"), []),
        "sample_titles": [t for t in (props.get("sample_titles_text") or "").split("\n") if t.strip()],
        "sample_chunks": [c for c in (props.get("sample_chunks_text") or "").split("\n\n") if c.strip()],
        "top_terms": [t.strip() for t in (props.get("top_terms_text") or "").split(",") if t.strip()],
        "_term_counts": _safe_json(props.get("term_counts_json"), {}),
        "updated_at": props.get("updated_at", ""),
    }


def upsert_collection_profile(client, profile: Dict[str, Any]) -> None:
    """
    Persiste (inserisce o aggiorna) il profilo di una collection nella
    collection tecnica ELYSIA_COLLECTION_PROFILES in Weaviate.

    Esegue un upsert: cerca l'oggetto esistente per collection_name,
    lo cancella e inserisce quello nuovo (Weaviate v4 non ha update nativo comodo).
    """
    try:
        _ensure_profiles_collection(client)
        col = client.collections.get(_PROFILES_COLLECTION_NAME)
        cname = profile["collection_name"]

        # Cerca oggetto esistente per questo collection_name
        result = col.query.fetch_objects(
            filters=Filter.by_property("collection_name").equal(cname),
            limit=1,
        )
        if result.objects:
            existing_uuid = result.objects[0].uuid
            col.data.delete_by_id(existing_uuid)
            print(f"[Profiles] Vecchio profilo '{cname}' rimosso (uuid={existing_uuid})")

        col.data.insert(properties=_profile_to_weaviate_properties(profile))
        print(f"[Profiles] Profilo '{cname}' salvato in Weaviate "
              f"({profile.get('document_count')} doc, {profile.get('chunk_count')} chunk).")

    except Exception as e:
        print(f"[Profiles] ERRORE upsert profilo '{profile.get('collection_name')}': {e}")
        traceback.print_exc()


def read_collection_profile_from_weaviate(client, collection_name: str) -> Dict[str, Any] | None:
    """
    Legge il profilo di una collection da Weaviate.
    Ritorna None se non trovato.
    """
    try:
        existing = client.collections.list_all()
        if _PROFILES_COLLECTION_NAME not in existing:
            return None
        col = client.collections.get(_PROFILES_COLLECTION_NAME)
        result = col.query.fetch_objects(
            filters=Filter.by_property("collection_name").equal(collection_name),
            limit=1,
        )
        if not result.objects:
            return None
        return _weaviate_properties_to_profile(result.objects[0].properties)
    except Exception as e:
        print(f"[Profiles] Errore lettura profilo '{collection_name}' da Weaviate: {e}")
        return None

# Assicurati che il processo Python veda Tesseract
os.environ["PATH"] = "/usr/local/bin:" + os.environ.get("PATH", "")


def _build_pdf_converter(*, do_ocr: bool) -> DocumentConverter:
    po = PdfPipelineOptions()
    po.do_ocr = do_ocr

    if do_ocr:
        # OCR sì, ma NON forzare full-page (quello ti ammazza i tempi)
        po.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=False)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)}
    )


# Due converter: uno veloce (no OCR) e uno fallback (OCR)
doc_converter_no_ocr = _build_pdf_converter(do_ocr=False)
doc_converter_ocr = _build_pdf_converter(do_ocr=True)


# -----------------------------
# Helpers
# -----------------------------
def preprocess_image_for_ocr(image_path: str) -> str:
    """
    Preprocessa un'immagine per migliorare la qualità dell'OCR:
    - Conversione in scala di grigi
    - Aumento del contrasto
    - Sharpening

    Args:
        image_path: Percorso dell'immagine da preprocessare

    Returns:
        Percorso dell'immagine preprocessata (stesso file sovrascritto)
    """
    try:
        print(f"[Preprocessing] Inizio preprocessing immagine: {image_path}")

        # Apri immagine
        img = Image.open(image_path)

        # 1. Conversione in scala di grigi
        if img.mode != 'L':
            img = img.convert('L')
            print(f"[Preprocessing] Convertita in scala di grigi")

        # 2. Aumento del contrasto (fattore 2.0 = raddoppia il contrasto)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        print(f"[Preprocessing] Contrasto aumentato (fattore 2.0)")

        # 3. Sharpening per migliorare nitidezza
        img = img.filter(ImageFilter.SHARPEN)
        print(f"[Preprocessing] Applicato sharpening")

        # Salva immagine preprocessata sovrascrivendo l'originale
        img.save(image_path)
        print(f"[Preprocessing] Immagine preprocessata salvata: {image_path}")

        return image_path

    except Exception as e:
        print(f"[Preprocessing] Errore durante preprocessing: {e}")
        # In caso di errore, ritorna il path originale senza preprocessing
        return image_path


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


def _is_running_in_docker() -> bool:
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")

def _ollama_reachable(url: str, timeout: float = 0.6) -> bool:
    try:
        r = requests.get(f"{url}/api/tags", timeout=timeout)
        return r.ok
    except Exception:
        return False


def _auto_detect_host_ollama_url(default: str = "http://localhost:11434") -> str:
    """
    Decide l'endpoint Ollama raggiungibile DAL PROCESSO PYTHON (host-side).
    - macOS tipico: http://localhost:11434
    - Linux aziendale (tuo compose): http://localhost:11435
    Override possibile via env OLLAMA_HOST.
    """
    env = os.environ.get("OLLAMA_HOST")
    if env:
        return env.rstrip("/")

    # 1) prova default 11434
    if _ollama_reachable("http://localhost:11434"):
        return "http://localhost:11434"

    # 2) prova la tua porta “anti-conflitto”
    if _ollama_reachable("http://localhost:11435"):
        return "http://localhost:11435"

    # fallback
    return default.rstrip("/")


def _normalize_ollama_endpoint(ollama_endpoint: str) -> str:
    # endpoint per PYTHON (host-side)
    if not ollama_endpoint:
        ollama_endpoint = _auto_detect_host_ollama_url()

    ollama_endpoint = ollama_endpoint.rstrip("/")
    cached = _OLLAMA_ENDPOINT_CACHE.get(ollama_endpoint)
    if cached:
        return cached

    # se è già un host non-local, non tocchiamo
    parsed = urlparse(ollama_endpoint)
    if parsed.hostname not in ("localhost", "127.0.0.1"):
        _OLLAMA_ENDPOINT_CACHE[ollama_endpoint] = ollama_endpoint
        return ollama_endpoint

    # se è localhost, lasciamo com'è (è proprio quello che serve al python)
    _OLLAMA_ENDPOINT_CACHE[ollama_endpoint] = ollama_endpoint
    return ollama_endpoint



def _normalize_ollama_endpoint_for_weaviate(url: str) -> str:
    """
    Endpoint che viene SALVATO nello schema Weaviate.
    Deve essere raggiungibile DA WEAVIATE (container), non dal python.
    Regole:
    - Se url è già non-localhost -> lo usiamo (scelta esplicita)
    - macOS/Windows: host.docker.internal:<port>
    - Linux:
        - se l'ollama host-side è 11435 (mappatura docker) -> in genere ollama è un servizio nel compose:
          quindi endpoint migliore per Weaviate è http://ollama:11434
        - altrimenti fallback su gateway docker (172.17.0.1:<port>)
    Override possibile via env WEAVIATE_OLLAMA_ENDPOINT
    """
    override = os.environ.get("WEAVIATE_OLLAMA_ENDPOINT")
    if override:
        return override.rstrip("/")

    if not url:
        url = _auto_detect_host_ollama_url()

    url = url.rstrip("/")
    cached = _OLLAMA_ENDPOINT_CACHE.get(f"weaviate::{url}")
    if cached:
        return cached

    parsed = urlparse(url)

    # scelta esplicita non-localhost: ok così
    if parsed.hostname not in ("localhost", "127.0.0.1"):
        _OLLAMA_ENDPOINT_CACHE[f"weaviate::{url}"] = url
        return url

    # port host-side (11434 su mac, 11435 su linux nel tuo caso)
    host_port = parsed.port or 11434
    system = platform.system().lower()

    # macOS / Windows: Weaviate container raggiunge l'host con host.docker.internal
    if system in ("darwin", "windows"):
        normalized = f"http://host.docker.internal:{host_port}"
        _OLLAMA_ENDPOINT_CACHE[f"weaviate::{url}"] = normalized
        return normalized

    # Linux: se stai usando la porta 11435, quasi sicuramente Ollama è nel docker-compose
    # e Weaviate deve chiamare il servizio 'ollama:11434'
    if host_port == 11435:
        normalized = "http://ollama:11434"
        _OLLAMA_ENDPOINT_CACHE[f"weaviate::{url}"] = normalized
        return normalized

    # fallback Linux: gateway docker verso host
    gateway_ip = os.environ.get("DOCKER_GATEWAY_IP", "172.17.0.1")
    normalized = f"http://{gateway_ip}:{host_port}"
    _OLLAMA_ENDPOINT_CACHE[f"weaviate::{url}"] = normalized
    return normalized


def get_ollama_embedding(text: str, ollama_url: str, model: str) -> list[float]:
    """
    Genera embedding tramite Ollama.
    """
    ollama_url = _normalize_ollama_endpoint(ollama_url)
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


def get_ollama_embeddings_batch(
        texts: List[str],
        ollama_url: str,
        model: str,
        batch_size: int = 10,
        progress_callback=None
) -> List[list[float]]:
    """
    Genera embedding in batch per velocizzare il processing.

    Args:
        texts: Lista di testi da cui generare embedding
        ollama_url: URL del server Ollama
        model: Nome del modello da usare
        batch_size: Numero di testi da processare per batch
        progress_callback: Funzione chiamata ad ogni batch completato (batch_num, total_batches)

    Returns:
        Lista di embedding (uno per testo)
    """
    ollama_url = _normalize_ollama_endpoint(ollama_url)
    embeddings = []
    total = len(texts)

    print(f"[Batch Embedding] Starting batch processing: {total} texts, batch_size={batch_size}")

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"[Batch Embedding] Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

        # Prova prima con batch API se supportata
        try:
            payload = {"model": model, "input": batch}
            resp = requests.post(f"{ollama_url}/api/embed", json=payload, timeout=180)
            if resp.ok:
                data = resp.json()
                batch_embeddings = data.get("embeddings", [])
                if len(batch_embeddings) == len(batch):
                    embeddings.extend(batch_embeddings)
                    print(f"[Batch Embedding] Batch {batch_num} OK (native batch API)")
                    # Callback DOPO il completamento del batch
                    if progress_callback:
                        progress_callback(batch_num, total_batches)
                    continue
        except Exception as e:
            print(f"[Batch Embedding] Native batch API failed: {e}, falling back to sequential")

        # Fallback: processa sequenzialmente il batch
        for idx, text in enumerate(batch):
            try:
                payload = {"model": model, "prompt": text}
                resp = requests.post(f"{ollama_url}/api/embeddings", json=payload, timeout=120)
                if resp.ok:
                    data = resp.json()
                    embeddings.append(data["embedding"])
                else:
                    raise Exception(f"Ollama error: {resp.status_code}")
            except Exception as e:
                print(f"[Batch Embedding] Error on text {i + idx}: {e}")
                raise

        print(f"[Batch Embedding] Batch {batch_num} completed (sequential fallback)")

        # Callback DOPO il completamento del batch (anche per fallback sequenziale)
        if progress_callback:
            progress_callback(batch_num, total_batches)

    print(f"[Batch Embedding] All batches completed: {len(embeddings)} embeddings generated")
    return embeddings


def doc_to_clean_text(doc) -> str:
    full_text_local = ""

    # Metodo 1: export_to_text
    if hasattr(doc, "export_to_text"):
        try:
            full_text_local = doc.export_to_text()
            print(f"[Docling] Used export_to_text: {len(full_text_local)} chars")
        except Exception as e:
            print(f"[Docling] export_to_text failed: {e}")

    # Metodo 2: iterate_items
    if not full_text_local or len(full_text_local.strip()) < 10:
        try:
            text_parts = []
            for item in doc.iterate_items():
                if isinstance(item, tuple):
                    item = item[0]
                if hasattr(item, "text") and item.text:
                    clean_text = normalize_text(item.text).strip()
                    if clean_text:
                        text_parts.append(clean_text)
            full_text_local = "\n\n".join(text_parts)
            print(f"[Docling] Used iterate_items: {len(text_parts)} elements, {len(full_text_local)} chars")
        except Exception as e:
            print(f"[Docling] iterate_items failed: {e}")

    # Metodo 3: export_to_markdown + pulizia
    if not full_text_local or len(full_text_local.strip()) < 10:
        try:
            md_text = doc.export_to_markdown()
            # Rimuove header markdown (# ## ###)
            full_text_local = re.sub(r"^#+\s*", "", md_text, flags=re.MULTILINE)
            # Rimuove bold (**testo**)
            full_text_local = re.sub(r"\*\*([^*]+)\*\*", r"\1", full_text_local)
            # Rimuove italic (*testo*)
            full_text_local = re.sub(r"\*([^*]+)\*", r"\1", full_text_local)
            # Rimuove code inline (`testo`)
            full_text_local = re.sub(r"`([^`]+)`", r"\1", full_text_local)
            # Rimuove link markdown [testo](url)
            full_text_local = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", full_text_local)
            print(f"[Docling] Used export_to_markdown (cleaned): {len(full_text_local)} chars")
        except Exception as e:
            print(f"[Docling] export_to_markdown failed: {e}")

    return full_text_local


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
        # Gestione file immagine: preprocessing prima dell'OCR
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp')):
            print(f"[Docling] Immagine rilevata, applicando preprocessing per OCR...")
            # Preprocessa immagine per migliorare OCR
            preprocess_image_for_ocr(file_path)

            # Ora procedi con Docling OCR sul file preprocessato
            t0 = time.perf_counter()
            result = doc_converter_ocr.convert(file_path)
            doc = result.document
            print(f"[Docling] convert(ocr) per immagine took {time.perf_counter() - t0:.2f}s")

            full_text = doc_to_clean_text(doc)
            print(f"[Docling] Final extracted text from image: {len(full_text)} chars")
        else:
            # ---- PASS 1: NO OCR ----
            t0 = time.perf_counter()
            result = doc_converter_no_ocr.convert(file_path)
            doc = result.document
            print(f"[Docling] convert(no_ocr) took {time.perf_counter() - t0:.2f}s")

            full_text = doc_to_clean_text(doc)
            print(f"[Docling] Final extracted text (no_ocr): {len(full_text)} chars")

            # Soglia: sotto qui consideri “testo insufficiente” e fai OCR
            MIN_CHARS_BEFORE_OCR = 2000

            # ---- PASS 2: OCR fallback (solo se serve e SOLO per PDF) ----
            if file_path.lower().endswith(".pdf"):
                if not full_text or len(full_text.strip()) < MIN_CHARS_BEFORE_OCR:
                    print(f"[Docling] Low text ({len(full_text.strip()) if full_text else 0} chars) -> OCR fallback...")

                    t1 = time.perf_counter()
                    result = doc_converter_ocr.convert(file_path)
                    doc = result.document
                    print(f"[Docling] convert(ocr) took {time.perf_counter() - t1:.2f}s")

                    full_text = doc_to_clean_text(doc)
                    print(f"[Docling] Final extracted text (ocr): {len(full_text)} chars")

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
    client = None
    try:
        data = request.get_json(force=True)
        collection_name = data['name']
        config = data['config']
        description = (data.get('description') or "").strip()
        force_recreate = bool(data.get('force_recreate', False))
        auto_recreate = bool(data.get('auto_recreate', True))

        print(f"[create-collection] name={collection_name}")
        print(f"[create-collection] config={config}")
        print(f"[create-collection] force_recreate={force_recreate}")
        print(f"[create-collection] auto_recreate={auto_recreate}")

        client = init_weaviate_client(
            config['weaviateHost'],
            config['weaviatePort']
        )

        status = _ensure_collection_ready(
            client,
            collection_name,
            config,
            description=description,
            force_recreate=force_recreate,
            auto_recreate=auto_recreate,
        )

        if status == "exists":
            return jsonify({'error': 'Collection already exists'}), 400

        return jsonify({'success': True, 'collection': collection_name, 'status': status})

    except Exception as e:
        print(f"[create-collection] ERROR: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if client is not None:
            client.close()


@app.route('/api/delete-collection', methods=['POST'])
def delete_collection():
    """Elimina una collezione Weaviate (schema + dati)."""
    client = None
    try:
        data = request.get_json(force=True)
        collection_name = data["name"]

        client = init_weaviate_client("127.0.0.1", "8080")

        existing = client.collections.list_all()
        if collection_name not in existing:
            return jsonify({"error": "Collection not found"}), 404

        client.collections.delete(collection_name)

        return jsonify({"success": True, "collection": collection_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if client is not None:
            client.close()


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

        # NON normalizziamo ollamaUrl qui perché il backend Flask gira sull'host
        # e deve usare localhost:11434 direttamente, non host.docker.internal
        # Solo Weaviate (in Docker) ha bisogno di http://ollama:11434

        # Auto-ricrea se endpoint/stopwords non sono corretti
        auto_recreate = bool(config.get('auto_recreate', True))
        if auto_recreate:
            client = init_weaviate_client(config['weaviateHost'], config['weaviatePort'])
            try:
                _ensure_collection_ready(
                    client,
                    collection_name,
                    config,
                    force_recreate=False,
                    auto_recreate=True,
                )
            finally:
                client.close()

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'Nessun file caricato'}), 400

        total_files = len(files)

        # Salva i file temporaneamente
        temp_files = []
        for file in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
            file.save(tmp.name)
            temp_files.append({
                'path': tmp.name,
                'filename': file.filename
            })
            tmp.close()

        # Progress init
        if upload_id:
            upload_progress[upload_id] = {
                "stage": "starting",
                "current": 0,
                "total": total_files,
                "percent": 0,
                "currentFile": "",
                "cancelled": False
            }

            UPLOAD_SESSIONS[upload_id] = {
                "collection": collection_name,
                "weaviateHost": config['weaviateHost'],
                "weaviatePort": config['weaviatePort'],
                "file_ids": [],
            }

        # Avvia processing in background
        thread = threading.Thread(
            target=process_files_docling_background,
            args=(upload_id, collection_name, config, temp_files, total_files)
        )
        thread.daemon = True
        thread.start()

        # Rispondi immediatamente
        return jsonify({
            "success": True,
            "message": "Upload avviato",
            "uploadId": upload_id
        })

    except Exception as e:
        # Progress error
        if upload_id:
            upload_progress[upload_id] = {
                "stage": "error",
                "current": 0,
                "total": 0,
                "percent": 0,
                "error": str(e),
                "cancelled": False
            }
        return jsonify({'error': str(e)}), 500


def process_files_docling_background(upload_id, collection_name, config, temp_files, total_files):
    """Processa i file in background usando Docling"""
    try:
        # Tokenizer coerente con modello embedding
        embed_model = config["embedModel"]
        tokenizer = get_tokenizer(embed_model)

        client = init_weaviate_client(config['weaviateHost'], config['weaviatePort'])
        collection = client.collections.get(collection_name)

        print(f"[Upload] Target: {config.get('weaviateHost')}:{config.get('weaviatePort')}")

        processed_files = 0
        uploaded_files = []
        failed_files = []

        # Carica profilo precedente (da cache in memoria o da Weaviate) per il seeding del term counter
        existing_profile = COLLECTION_PROFILES.get(collection_name) or \
                           read_collection_profile_from_weaviate(client, collection_name)
        seed_counts = existing_profile.get("_term_counts", {}) if existing_profile else {}

        # Builder per il profilo della collection (con seed dal profilo precedente se presente)
        profile_builder = CollectionProfileBuilder(
            collection_name=collection_name,
            searchable_fields=_COLLECTION_SEARCHABLE_FIELDS,
            content_fields=_COLLECTION_CONTENT_FIELDS,
            seed_term_counts=seed_counts,
        )

        # Formati supportati da Docling
        supported_extensions = [
            '.pdf', '.docx', '.pptx', '.html', '.md', '.txt',
            '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'
        ]

        for file_idx, temp_file in enumerate(temp_files):
            tmp_path = temp_file['path']
            filename = temp_file['filename']

            print(f"[UPLOAD] Inizio elaborazione file {file_idx + 1}/{total_files}: {filename}")

            # CONTROLLA SE UPLOAD È STATO CANCELLATO
            if upload_id and upload_id in upload_progress:
                if upload_progress[upload_id].get("cancelled", False):
                    print(f"[Upload] Upload {upload_id} cancellato dall'utente, interrompo...")
                    break

            file_id = str(uuid.uuid4())
            title = Path(filename).stem

            # Progress: file corrente
            if upload_id:
                upload_progress[upload_id] = {
                    "stage": "processing",
                    "current": file_idx,
                    "total": total_files,
                    "percent": int((file_idx / total_files) * 100),
                    "currentFile": filename,
                    "cancelled": False
                }

            try:
                # Verifica formato supportato
                if not any(filename.lower().endswith(ext) for ext in supported_extensions):
                    failed_files.append({
                        "filename": filename,
                        "error": "Formato non supportato (usa PDF, DOCX, PPTX, HTML, TXT o immagini)",
                    })
                    continue

                doc_type = Path(filename).suffix[1:]

                # Ottieni preset chunking (default: balanced)
                preset_name = config.get('ingestPreset', 'balanced')
                preset = INGEST_PRESETS.get(preset_name, INGEST_PRESETS['balanced'])

                print(
                    f"[Upload] Using chunking preset: {preset_name} (max_tokens={preset['max_tokens']}, overlap={preset['overlap_tokens']})")

                # Estrazione con Docling + chunking intelligente con overlap
                print(f"[UPLOAD] Estrazione e chunking del file: {filename}")
                blocks = extract_with_docling(
                    tmp_path,
                    tokenizer=tokenizer,
                    max_tokens=preset['max_tokens'],
                    overlap_tokens=preset['overlap_tokens']
                )

                print(f"[UPLOAD] Chunking completato: {len(blocks)} chunk estratti")

                if not blocks:
                    failed_files.append({
                        "filename": filename,
                        "error": "Nessun contenuto estratto (file vuoto o corrotto?)",
                    })
                    continue

                # DIAGNOSTICA: Statistiche sui chunk generati
                total_chunks = len(blocks)
                chunk_sizes = [len(b.get("content", "")) for b in blocks]
                avg_chunk_size = sum(chunk_sizes) / total_chunks if total_chunks > 0 else 0
                min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
                max_chunk_size = max(chunk_sizes) if chunk_sizes else 0

                print(f"[Upload] CHUNK STATS - File: {filename}")
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

                print(f"[Upload] File: {filename} - Chunks: {len(blocks)} - Lang: {detected_lang}")

                # AGGIUNGI file_id SUBITO alla sessione
                if upload_id and upload_id in UPLOAD_SESSIONS:
                    UPLOAD_SESSIONS[upload_id]["file_ids"].append(file_id)

                total_chunks = len(blocks)

                # Step 1: Estrai tutti i testi dai chunk
                chunk_texts = [block["content"] for block in blocks]

                # Step 2: Controlla cancellazione prima del batch embedding
                if upload_id and upload_id in upload_progress:
                    if upload_progress[upload_id].get("cancelled", False):
                        print(f"[Upload] Upload {upload_id} cancellato prima del batch embedding")
                        raise Exception("Upload cancellato dall'utente")

                # Step 3: Callback per aggiornare progress durante batch embedding
                def batch_progress_callback(batch_num, total_batches):
                    if upload_id:
                        # Calcola progress: 10% base + (40% * progresso batch)
                        batch_progress = batch_num / total_batches
                        file_progress = 0.1 + (batch_progress * 0.4)  # 10% - 50%
                        overall_progress = (file_idx + file_progress) / total_files
                        upload_progress[upload_id] = {
                            "stage": "processing",
                            "current": file_idx,
                            "total": total_files,
                            "percent": int(overall_progress * 100),
                            "currentFile": f"{filename} (embedding batch {batch_num}/{total_batches})",
                            "cancelled": False
                        }

                # Step 4: Genera tutti gli embedding in batch
                print(f"[UPLOAD] Inizio batch embedding per {len(blocks)} chunk...")
                embeddings = get_ollama_embeddings_batch(
                    chunk_texts,
                    config['ollamaUrl'],
                    config['embedModel'],
                    batch_size=10,
                    progress_callback=batch_progress_callback
                )
                print(f"[UPLOAD] Batch embedding completato: {len(embeddings)} embedding generati")

                # Step 5: Controlla cancellazione dopo batch embedding
                if upload_id and upload_id in upload_progress:
                    if upload_progress[upload_id].get("cancelled", False):
                        print(f"[Upload] Upload {upload_id} cancellato dopo batch embedding")
                        raise Exception("Upload cancellato dall'utente")

                # Step 6: Progress: inizio inserimento parallelo
                if upload_id:
                    file_progress = 0.5
                    overall_progress = (file_idx + file_progress) / total_files
                    upload_progress[upload_id] = {
                        "stage": "processing",
                        "current": file_idx,
                        "total": total_files,
                        "percent": int(overall_progress * 100),
                        "currentFile": f"{filename} (inserting chunks...)",
                        "cancelled": False
                    }

                # Step 7: Funzione per inserire un singolo chunk
                def insert_chunk(idx, block, embedding_vec):
                    """Inserisce un chunk in Weaviate (eseguito in parallelo)"""
                    # Controlla cancellazione anche nei thread
                    if upload_id and upload_id in upload_progress:
                        if upload_progress[upload_id].get("cancelled", False):
                            raise Exception("Upload cancellato dall'utente")

                    result = collection.data.insert(
                        properties={
                            "title": title,
                            "file_id": file_id,
                            "chunk_index": idx,
                            "doc_type": doc_type,
                            "content": block["content"],
                            "lang": detected_lang,
                            "page": block.get("page", 0),
                            "block_kind": block.get("kind", "docling_chunk"),
                        },
                        vector=embedding_vec,
                    )
                    return idx, result

                # Step 8: Inserisci chunk in parallelo
                print(f"[UPLOAD] Inizio inserimento parallelo dei chunk in Weaviate...")
                inserted_count = 0

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    for idx, (block, embedding_vec) in enumerate(zip(blocks, embeddings)):
                        future = executor.submit(insert_chunk, idx, block, embedding_vec)
                        futures.append(future)

                    for future in as_completed(futures):
                        try:
                            if upload_id and upload_id in upload_progress:
                                if upload_progress[upload_id].get("cancelled", False):
                                    print(f"[Upload] Upload {upload_id} cancellato durante inserimento parallelo")
                                    executor.shutdown(wait=False, cancel_futures=True)
                                    raise Exception("Upload cancellato dall'utente")

                            idx, result = future.result()
                            inserted_count += 1
                            # AGGIORNA PROGRESS con contatore chunk come in app.py
                            if upload_id:
                                insert_progress = inserted_count / total_chunks
                                file_progress = 0.5 + (insert_progress * 0.5)
                                overall_progress = (file_idx + file_progress) / total_files
                                upload_progress[upload_id] = {
                                    "stage": "processing",
                                    "current": file_idx,
                                    "total": total_files,
                                    "percent": int(overall_progress * 100),
                                    "currentFile": f"{filename} (chunk {inserted_count}/{total_chunks})",
                                    "cancelled": False
                                }
                            if inserted_count % 10 == 0 or inserted_count == total_chunks:
                                print(f"[UPLOAD] {inserted_count}/{total_chunks} chunk inseriti in Weaviate")
                        except Exception as e:
                            print(f"[UPLOAD] Errore durante inserimento chunk: {e}")
                            raise

                print(f"[UPLOAD] Inserimento completato: {inserted_count} chunk inseriti per il file {filename}")

                # Aggiorna il profilo della collection con i dati di questo documento
                profile_builder.add_document(
                    title=title,
                    doc_type=doc_type,
                    detected_lang=detected_lang,
                    chunks=chunk_texts,
                )

                processed_files += 1
                uploaded_files.append(filename)

            except Exception as file_error:
                if "Upload cancellato" in str(file_error):
                    print(f"[Upload] Cancellazione rilevata, interrompo loop...")
                    break

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
                    "filename": filename,
                    "error": friendly_msg,
                })

            finally:
                os.unlink(tmp_path)

        # Costruisce e salva il profilo della collection (anche in caso di upload parziale)
        if processed_files > 0:
            new_profile = profile_builder.build()

            # Legge il profilo esistente: prima in memoria, poi da Weaviate
            existing = COLLECTION_PROFILES.get(collection_name)
            if existing is None:
                existing = read_collection_profile_from_weaviate(client, collection_name)

            if existing:
                # --- Merge rules esplicite ---

                # Contatori scalari: somma
                new_profile["document_count"] += existing.get("document_count", 0)
                new_profile["chunk_count"] += existing.get("chunk_count", 0)

                # Distribuzioni: merge con somma per chiave
                for lang, cnt in existing.get("languages_distribution", {}).items():
                    new_profile["languages_distribution"][lang] = (
                        new_profile["languages_distribution"].get(lang, 0) + cnt
                    )
                for dt, cnt in existing.get("doc_types_distribution", {}).items():
                    new_profile["doc_types_distribution"][dt] = (
                        new_profile["doc_types_distribution"].get(dt, 0) + cnt
                    )

                # sample_titles: unione deduplicata con limite
                merged_titles = new_profile["sample_titles"] + [
                    t for t in existing.get("sample_titles", [])
                    if t not in new_profile["sample_titles"]
                ]
                new_profile["sample_titles"] = merged_titles[:CollectionProfileBuilder.MAX_SAMPLE_TITLES]

                # sample_chunks: unione deduplicata con limite (trim già fatto nel builder)
                merged_chunks = new_profile["sample_chunks"] + [
                    c for c in existing.get("sample_chunks", [])
                    if c not in new_profile["sample_chunks"]
                ]
                new_profile["sample_chunks"] = merged_chunks[:CollectionProfileBuilder.MAX_SAMPLE_CHUNKS]

                # top_terms: ricalcolo dal contatore grezzo unito (non merge di liste finali)
                merged_counts = collections.Counter(existing.get("_term_counts", {}))
                merged_counts.update(new_profile.get("_term_counts", {}))
                new_profile["_term_counts"] = dict(merged_counts)
                new_profile["top_terms"] = [
                    term for term, _ in merged_counts.most_common(CollectionProfileBuilder.TOP_TERMS_N)
                ]

                # updated_at: timestamp più recente (quello appena calcolato)
                # già impostato da build(), nessuna azione necessaria

                # Ricalcola description_text con i dati aggiornati
                new_profile["description_text"] = CollectionProfileBuilder.build_description_text(
                    collection_name=collection_name,
                    document_count=new_profile["document_count"],
                    chunk_count=new_profile["chunk_count"],
                    languages_distribution=new_profile["languages_distribution"],
                    doc_types_distribution=new_profile["doc_types_distribution"],
                    searchable_fields=new_profile["searchable_fields"],
                    top_terms=new_profile["top_terms"],
                    sample_titles=new_profile["sample_titles"],
                )

            # Salva in memoria
            COLLECTION_PROFILES[collection_name] = new_profile

            # Persisti su Weaviate (upsert nella collection tecnica)
            upsert_collection_profile(client, new_profile)

            print(
                f"[UPLOAD] Profilo collection '{collection_name}' aggiornato: "
                f"{new_profile['document_count']} doc, {new_profile['chunk_count']} chunk, "
                f"top_terms={new_profile['top_terms'][:5]}"
            )

        # Progress done (o cancelled se interrotto)
        if upload_id:
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
                print(f"[UPLOAD] Upload {upload_id} completamente cancellato")
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

        print(f"[UPLOAD] Background processing completato:")
        print(f"  - Processati: {processed_files}")
        print(f"  - Caricati con successo: {uploaded_files}")
        print(f"  - Falliti: {failed_files}")
        print(f"  - Collection: {collection_name}")

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
        traceback.print_exc()
        print(f"[UPLOAD ERROR] {str(e)}")


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

        upload_progress.pop(upload_id, None)
        UPLOAD_SESSIONS.pop(upload_id, None)

        print(f"[Cancel] Upload {upload_id} cancellato, eliminati {deleted} file")
        return jsonify({"success": True, "deleted": deleted})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/collection-profile/<collection_name>', methods=['GET'])
def get_collection_profile(collection_name: str):
    """
    Ritorna il profilo aggregato di una collection.
    Cerca prima in memoria (COLLECTION_PROFILES), poi in Weaviate come fallback.
    Query params opzionali: host, port (default 127.0.0.1:8080)
    """
    # 1. Cache in memoria
    profile = COLLECTION_PROFILES.get(collection_name)
    if profile:
        return jsonify(profile)

    # 2. Fallback: leggi da Weaviate
    host = request.args.get('host', '127.0.0.1')
    port = request.args.get('port', '8080')
    client = None
    try:
        client = init_weaviate_client(host, port)
        profile = read_collection_profile_from_weaviate(client, collection_name)
    except Exception as e:
        print(f"[Profiles] Errore connessione Weaviate per lettura profilo: {e}")
    finally:
        if client is not None:
            client.close()

    if not profile:
        return jsonify({"error": f"Nessun profilo disponibile per la collection '{collection_name}'"}), 404

    # Carica in memoria per richieste future
    COLLECTION_PROFILES[collection_name] = profile
    return jsonify(profile)


@app.route('/api/collection-profiles', methods=['GET'])
def get_all_collection_profiles():
    """
    Ritorna tutti i profili disponibili.
    Legge prima tutti i profili da Weaviate (fonte di verità),
    poi integra con quelli in memoria non ancora persistiti.
    Query params opzionali: host, port (default 127.0.0.1:8080)
    """
    host = request.args.get('host', '127.0.0.1')
    port = request.args.get('port', '8080')
    client = None
    weaviate_profiles: Dict[str, Dict] = {}

    try:
        client = init_weaviate_client(host, port)
        existing = client.collections.list_all()
        if _PROFILES_COLLECTION_NAME in existing:
            col = client.collections.get(_PROFILES_COLLECTION_NAME)
            result = col.query.fetch_objects(limit=200)
            for obj in result.objects:
                p = _weaviate_properties_to_profile(obj.properties)
                weaviate_profiles[p["collection_name"]] = p
    except Exception as e:
        print(f"[Profiles] Errore lettura profili da Weaviate: {e}")
    finally:
        if client is not None:
            client.close()

    # Unisci con cache in memoria (priorità a Weaviate)
    merged: Dict[str, Dict] = {**COLLECTION_PROFILES, **weaviate_profiles}
    return jsonify(list(merged.values()))


@app.route('/api/collections', methods=['GET'])
def get_collections():
    """Ottieni lista collezioni"""
    client = None
    try:
        host = request.args.get('host', '127.0.0.1')
        port = request.args.get('port', '8080')

        client = init_weaviate_client(host, port)
        collections = client.collections.list_all()

        return jsonify({'collections': [{'name': col} for col in collections]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if client is not None:
            client.close()


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'ok'})


@app.route("/api/debug-count/<collection_name>", methods=["GET"])
def debug_count(collection_name):
    host = request.args.get("host", "127.0.0.1")
    port = request.args.get("port", "8080")
    client = None
    try:
        client = init_weaviate_client(host, port)
        col = client.collections.get(collection_name)

        print(f"[Debug] Counting collection: {collection_name}")

        res = col.aggregate.over_all(total_count=True)
        return jsonify({"collection": collection_name, "total_count": res.total_count})
    finally:
        if client is not None:
            client.close()


def _has_localhost_endpoint(cfg) -> bool:
    if cfg.vectorizer_config and cfg.vectorizer_config.model:
        endpoint = getattr(cfg.vectorizer_config.model, "api_endpoint", None)
        if endpoint and ("localhost" in endpoint or "127.0.0.1" in endpoint):
            return True
    if cfg.vector_config:
        for _, vec_cfg in cfg.vector_config.items():
            if hasattr(vec_cfg, "vectorizer") and vec_cfg.vectorizer:
                model = vec_cfg.vectorizer.model
                endpoint = getattr(model, "api_endpoint", None) if model else None
                if endpoint and ("localhost" in endpoint or "127.0.0.1" in endpoint):
                    return True
    return False


def _stopwords_disabled(cfg) -> bool:
    sw = cfg.inverted_index_config.stopwords if cfg.inverted_index_config else None
    if sw is None:
        return True
    removals = set(sw.removals or [])
    return {"the", "and", "is", "of", "to"}.issubset(removals)



def _create_collection(client, collection_name: str, config: dict, description: str = "") -> None:
    ollama_endpoint = _normalize_ollama_endpoint_for_weaviate(
        config.get("ollamaUrl", "http://localhost:11434")
    )

    vectorizer = Configure.Vectorizer.text2vec_ollama(
        model=config.get("embedModel", "qwen3-embedding:4b"),
        api_endpoint=ollama_endpoint,
        vectorize_collection_name=True,
    )

    inverted_index_kwargs = dict(
        index_null_state=True,
        index_property_length=True,
        index_timestamps=True,
        stopwords_removals=ENGLISH_STOPWORDS,
    )

    client.collections.create(
        name=collection_name,
        description=description or None,
        properties=[
            Property(name="title", data_type=DataType.TEXT,
                     description="Original document name"),
            Property(name="file_id", data_type=DataType.TEXT,
                     description="Internal unique id for this file (per upload)."),
            Property(name="chunk_index", data_type=DataType.INT,
                     description="Chunk index"),
            Property(name="doc_type", data_type=DataType.TEXT,
                     description="Document type"),
            Property(name="content", data_type=DataType.TEXT,
                     description="Document content (chunk)"),
            Property(name="lang", data_type=DataType.TEXT,
                     description="Detected language"),
            Property(name="page", data_type=DataType.INT,
                     description="Document page number"),
            Property(name="block_kind", data_type=DataType.TEXT,
                     description="Chunk type (docling_chunk)"),
        ],
        vectorizer_config=vectorizer,
        inverted_index_config=Configure.inverted_index(**inverted_index_kwargs),
    )


def _ensure_collection_ready(
    client,
    collection_name: str,
    config: dict,
    *,
    description: str = "",
    force_recreate: bool = False,
    auto_recreate: bool = True,
) -> str:
    existing_collections = client.collections.list_all()
    if collection_name in existing_collections:
        recreate_needed = False
        if auto_recreate:
            try:
                current_cfg = client.collections.get(collection_name).config.get()
                recreate_needed = (
                    _has_localhost_endpoint(current_cfg)
                    or not _stopwords_disabled(current_cfg)
                )
            except Exception:
                recreate_needed = True

        if not force_recreate and not recreate_needed:
            return "exists"

        client.collections.delete(collection_name)
        chunked_name = f"ELYSIA_CHUNKED_{collection_name.lower()}__"
        if chunked_name in existing_collections:
            client.collections.delete(chunked_name)

        _create_collection(client, collection_name, config, description)
        return "recreated"

    _create_collection(client, collection_name, config, description)
    return "created"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

