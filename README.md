# Weaviate Document Manager

Una dashboard per l'ingestione di documenti (RAG Pipeline) che gira interamente in locale.
Questo tool permette di caricare file PDF e TXT, suddividerli in chunk, generare embedding tramite **Ollama** e salvarli in un database vettoriale **Weaviate**.


## 📋 Prerequisiti

Prima di iniziare, assicurati di avere installato:

1.  **Node.js** (v18+ raccomandato) per il frontend.
2.  **Python** (v3.12) per il backend.
3.  **Docker** (per eseguire Weaviate).
4.  **Ollama** installato e funzionante in locale.

---

## 🚀 Setup Iniziale (Servizi Esterni)

Il progetto richiede che Weaviate e Ollama siano attivi **prima** di lanciare l'applicazione.

### 1. Configura Ollama
Il progetto è configurato di default per usare il modello `mxbai-embed-large`. Assicurati di averlo scaricato:

```bash
ollama pull mxbai-embed-large
```

### 2. Avvia Weaviate (via Docker)
Il progetto include già un file di configurazione per Weaviate. Avvialo semplicemente con:

```bash
docker-compose up -d
```

---

## 🛠️ Installazione e Avvio

Il progetto è diviso in due parti: Backend (Python) e Frontend (React). Dovrai avviare due terminali separati.

### 3. Backend (Flask API)

Apri un terminale e naviga nella cartella `backend`:

```bash
cd backend
```

Crea un ambiente virtuale (consigliato) e attivalo:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

Installa le dipendenze necessarie:
```bash
pip install -r requirements.txt
```

Avvia il server:
```bash
python app.py
```

Il backend sarà attivo su http://localhost:5001.

### 4. Frontend (Dashboard React)
Apri un nuovo terminale e naviga nella cartella `frontend`:

```bash
cd frontend
```

Installa le dipendenze:
```bash
npm install
```

Avvia l'interfaccia:
```bash
npm run dev
```
Apri il browser all'indirizzo mostrato (solitamente http://localhost:5173).