# GapAgent 

> An AI-powered academic research gap identification assistant built with Flask, ChromaDB, SentenceTransformers, and Ollama LLMs.

GapAgent helps researchers and students discover underexplored areas in academic literature by analyzing text, PDFs, and topic queries. It combines retrieval-augmented generation (RAG), PECO-based evidence extraction, and local LLM reasoning to surface prioritized, deduplicated research gaps with novelty scores.

---

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [API Reference](#api-reference)
- [System Overview](#system-overview)
  - [Core Pipeline](#core-pipeline)
  - [Models & Components](#models--components)
  - [Gap Classification](#gap-classification)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## Features

- **Three input modes** — paste raw text/abstracts, upload PDF papers, or search by topic
- **PECO extraction** — identifies Population, Exposure, Comparison, and Outcome from literature
- **RAG-powered retrieval** — ChromaDB + `all-MiniLM-L6-v2` embeddings for semantic context lookup
- **Dual-model LLM pipeline** — Mistral for core analysis, Llama3 for optional refinement
- **Gap scoring** — priority levels (High / Medium / Low) and novelty scores (0–100)
- **Deduplication** — removes redundant gaps using word-overlap similarity
- **Session memory** — tracks literature, evidence, and gaps across requests
- **Live topic search** — parallel ArXiv + Semantic Scholar queries

---

## System Requirements

| Requirement | Version |
|---|---|
| Python | 3.9 or higher |
| Ollama | Latest (local LLM runtime) |
| RAM | 8 GB minimum (16 GB recommended) |
| Disk | ~5 GB for models |

---

## Project Structure

```
gapagent/
├── app.py              # Main Flask application and analysis pipeline
├── index.html          # Browser-based frontend interface
├── requirements.txt    # Python dependencies
└── rag_db/             # ChromaDB persistent vector database (auto-created)
```

---

## Setup

### 1. Create Project Folder

```
GapAgent1/
├── app.py              
├── index.html         
├── requirements.txt    
└── readme.md             
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Start Ollama

Download Ollama from [https://ollama.com](https://ollama.com) and ensure it is running at `http://127.0.0.1:11434`.

Then pull the required models:

```bash
ollama pull mistral
ollama pull llama3
```

> **Tip for faster performance:** Use the quantized Mistral variant:
> ```bash
> ollama pull mistral:7b-instruct-q4_K_M
> ```

---

## Configuration

GapAgent can be tuned using environment variables before launch:

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_MODEL` | `mistral` | Ollama model used for core analysis |
| `LLAMA_MODEL` | `llama3` | Ollama model used for optional refinement |
| `SINGLE_MODEL_MODE` | `true` | Skip Llama3 pass to save ~45 seconds per analysis |
| `INPUT_CHAR_LIMIT` | `2000` | Max characters of text sent to the LLM |

**Example — full dual-model pipeline:**

```bash
export SINGLE_MODEL_MODE=false
export INPUT_CHAR_LIMIT=3000
python app.py
```

---

## Running the App

```bash
python app.py
```

The server starts at **http://localhost:8000**.

---

## Usage

### Web Interface

Open your browser and go to:

```
http://localhost:8000
```

The interface lets you:
- Paste text or abstracts for instant analysis
- Upload a PDF research paper
- Enter a topic to trigger a live literature search

### API Reference

All endpoints return JSON. The base URL is `http://localhost:8000`.

---

#### `GET /api/ping`
Health check. Returns app metadata and status.

```bash
curl http://localhost:8000/api/ping
```

---

#### `POST /api/analyze/text`
Analyze a raw text snippet or abstract.

**Request body:**
```json
{
  "title": "Paper Title (optional)",
  "text": "Your abstract or text here (min 50 characters)"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"title": "AI in Healthcare", "text": "This study explores..."}'
```

---

#### `POST /api/analyze/pdf`
Upload and analyze a PDF paper. Sends as `multipart/form-data`.

**Example:**
```bash
curl -X POST http://localhost:8000/api/analyze/pdf \
  -F "file=@paper.pdf"
```

> Note: Scanned/image-based PDFs are not supported (OCR is not included).

---

#### `POST /api/analyze/topic`
Search ArXiv and Semantic Scholar for a topic, store results in RAG, then analyze.

**Request body:**
```json
{
  "topic": "federated learning privacy"
}
```

---

#### `POST /api/chat`
Ask a follow-up question about analyzed content.

**Request body:**
```json
{
  "message": "What are the most critical gaps found?"
}
```

---

#### `GET /api/report`
Returns a Markdown-formatted research report plus session statistics.

```bash
curl http://localhost:8000/api/report
```

---

#### `GET /api/memory`
Returns the current session state: literature entries, extracted evidence, and identified gaps.

```bash
curl http://localhost:8000/api/memory
```

---

#### `POST /api/memory/clear`
Clears all session memory. Useful when starting a new research topic.

```bash
curl -X POST http://localhost:8000/api/memory/clear
```

---

## System Overview

### Core Pipeline

```
Input (text / PDF / topic)
        │
        ▼
  Text Extraction
  + Metadata Parsing
        │
        ▼
  Chunk & Store
  in ChromaDB (RAG)
        │
        ▼
  Semantic Retrieval
  (query → top-k chunks)
        │
        ▼
  PECO Extraction
  via Mistral
        │
        ▼
  Gap Analysis
  via Mistral (+ optional Llama3 refinement)
        │
        ▼
  Post-Processing:
  dedup · priority · novelty score · gap type
        │
        ▼
  Store in Memory
  + Return JSON
```

### Models & Components

| Component | Role |
|---|---|
| `MISTRAL_MODEL` | Core analysis — PECO extraction, gap identification |
| `LLAMA_MODEL` | Optional refinement pass for improved output quality |
| `all-MiniLM-L6-v2` | Sentence embeddings for RAG semantic search |
| `ChromaDB` | Persistent vector store for retrieved paper chunks |
| `Ollama API` | Local LLM inference runtime |
| `ArXiv API` | Live academic paper search |
| `Semantic Scholar API` | Live citation-aware paper search |

### Gap Classification

Each identified gap is automatically classified and scored:

**Priority levels** (based on keyword matching):

| Priority | Example Keywords |
|---|---|
| High | bias, privacy, ethics, no study, unexplored, critical |
| Medium | limited, insufficient, scalability, few studies |
| Low | minor, optional, suggested, future |

**Gap types** (from decision-tree categorization):

- Insufficient Research
- Poor Study Quality
- Inconsistent Findings
- Population Gap
- Lack of Causal Evidence
- Lack of Real-World Evidence
- General Research Gap

**Novelty score (0–100):** Computed by measuring word-overlap against already-identified gaps in the session. Higher scores indicate more distinct, previously unseen gaps.

---

## Troubleshooting

**Ollama not reachable**
Verify Ollama is running and accessible:
```bash
curl http://127.0.0.1:11434/api/tags
```

**PDF extraction returns empty text**
The file may be a scanned or image-based PDF. GapAgent does not include OCR support. Try a text-based PDF instead.

**Slow analysis**
Enable single-model mode to skip the Llama3 refinement pass:
```bash
export SINGLE_MODEL_MODE=true
```

**Stale session data affecting results**
Clear session memory between unrelated research topics:
```bash
curl -X POST http://localhost:8000/api/memory/clear
```

**ChromaDB errors on startup**
Delete the `rag_db/` folder and restart — it will be recreated automatically:
```bash
rm -rf rag_db/
python app.py
```

---

## Credits

Developed by **Onaiza Maryam** and **Iqra Rani**  
UET Taxila — AI Course Project

---

*GapAgent uses the PECO framework and decision-tree gap classification based on established academic evidence synthesis methodology.*
