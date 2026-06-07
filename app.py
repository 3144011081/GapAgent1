"""
GapAgent — A+ Final Version
Based on: Research Gap Identification Assistant paper
Authors: Onaiza Maryam (23-SE-87) & Iqra Rani (23-SE-60)
Course : Artificial Intelligence — Dr. Kanwal Yousaf, UET Taxila
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests, re, urllib.parse, io, os
from PyPDF2 import PdfReader
import feedparser
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# APP + CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app, resources={r"/*": {"origins": "*"}})

OLLAMA_URL    = "http://127.0.0.1:11434/api/generate"
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral")  # deep reasoning
LLAMA_MODEL   = os.getenv("LLAMA_MODEL",   "llama3")   # refinement

SYSTEM_PROMPT = (
    "You are GapAgent, an elite AI research analyst specialized in academic "
    "literature analysis and research gap identification. You use PECO framework, "
    "decision tree logic, and evidence-based reasoning to identify specific, novel, "
    "and actionable research gaps. Always be precise and academically rigorous. "
    "Never repeat the same point twice. Generate only unique, distinct insights."
)

# ─────────────────────────────────────────────
# PRIORITY KEYWORDS (from paper + review feedback)
# ─────────────────────────────────────────────
HIGH_KEYWORDS   = [
    "bias", "privacy", "security", "ethics", "fairness",
    "lack", "missing", "no study", "unexplored", "absent",
    "critical", "urgent", "fundamental", "significant gap",
]
MEDIUM_KEYWORDS = [
    "infrastructure", "training", "policy", "scalability",
    "limited", "insufficient", "few studies", "rarely",
    "moderate", "partial", "incomplete",
]
LOW_KEYWORDS    = [
    "minor", "slight", "marginal", "secondary",
    "optional", "future", "suggested", "could be",
]

# GAP TYPE CATEGORIES (from paper Figure 2 Decision Tree)
GAP_TYPE_KEYWORDS = {
    "Insufficient Research":        ["no study", "very few", "limited research", "lack of studies"],
    "Poor Study Quality":           ["bias", "low quality", "methodological", "unreliable"],
    "Inconsistent Findings":        ["contradict", "inconsistent", "conflicting", "disagree"],
    "Population Gap":               ["population", "subgroup", "demographic", "underrepresented"],
    "Lack of Causal Evidence":      ["causal", "causality", "correlation only", "association"],
    "Lack of Real-World Evidence":  ["real-world", "clinical", "deployment", "practical setting"],
}

# ─────────────────────────────────────────────
# RAG INIT
# ─────────────────────────────────────────────
print("🔧 Loading embedding model (all-MiniLM-L6-v2)...")
embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./rag_db")
collection    = chroma_client.get_or_create_collection(name="research_papers")
print("✅ RAG ready.")

# ─────────────────────────────────────────────
# MEMORY MODULE (Literature / Evidence / Gap Tables)
# ─────────────────────────────────────────────
memory = {
    "literature": [],  # {title, year, abstract}          ← Literature Table
    "evidence":   [],  # {paper, findings, limitation}     ← Evidence Table
    "gaps":       [],  # {gap, evidence, priority}         ← Gap Table
}

# ─────────────────────────────────────────────
# POST-PROCESSING LAYER (fixes review issues)
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Fix broken words, truncations, extra whitespace from LLM output."""
    if not text: return ""
    # Fix common LLM truncation artifacts
    fixes = {
        "biase":    "bias",
        "skill sh": "skill shortage",
        "infra ":   "infrastructure ",
        "privac":   "privacy",
        "algorith":  "algorithm",
    }
    for bad, good in fixes.items():
        text = text.replace(bad, good)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def remove_duplicates(items: list) -> list:
    """Remove duplicate strings — case insensitive, strip whitespace."""
    seen, clean = set(), []
    for item in items:
        key = item.lower().strip()[:120]  # compare first 120 chars
        if key and key not in seen:
            seen.add(key)
            clean.append(item)
    return clean


def assign_priority(text: str) -> str:
    """Smart priority scoring based on keyword matching."""
    t = text.lower()
    if any(k in t for k in HIGH_KEYWORDS):   return "High"
    if any(k in t for k in MEDIUM_KEYWORDS): return "Medium"
    if any(k in t for k in LOW_KEYWORDS):    return "Low"
    return "Medium"  # default


def assign_gap_type(text: str) -> str:
    """Classify gap into one of 6 types from paper's Decision Tree (Figure 2)."""
    t = text.lower()
    for gap_type, keywords in GAP_TYPE_KEYWORDS.items():
        if any(k in t for k in keywords):
            return gap_type
    return "General Research Gap"


def compute_novelty_score(gap: str, existing_gaps: list) -> int:
    """
    Novelty index 0–100.
    High score = very different from existing gaps.
    Low score  = similar to already identified gaps.
    """
    if not existing_gaps: return 85  # first gap is always novel
    gap_words = set(gap.lower().split())
    max_overlap = 0
    for existing in existing_gaps:
        existing_words = set(existing.lower().split())
        if not existing_words: continue
        overlap = len(gap_words & existing_words) / max(len(gap_words), len(existing_words))
        max_overlap = max(max_overlap, overlap)
    return max(10, int((1 - max_overlap) * 100))


def cluster_gaps(gaps: list) -> dict:
    """Group gaps by their gap_type for themed output."""
    clusters = {}
    for g in gaps:
        t = g.get("gap_type", "General")
        clusters.setdefault(t, []).append(g)
    return clusters


def validate_output(data: dict) -> dict:
    """
    Final validation pass — clean all text fields,
    deduplicate lists, ensure no empty values.
    """
    # Clean text fields
    for field in ["summary", "limitations", "future"]:
        if field in data:
            data[field] = clean_text(data[field])

    # Deduplicate and validate gaps
    if "gaps" in data:
        seen_gaps = []
        clean_gaps = []
        for g in data["gaps"]:
            gap_text = clean_text(g.get("gap", ""))
            if not gap_text or len(gap_text) < 15: continue
            # Check duplicate
            is_dup = any(
                len(set(gap_text.lower().split()) & set(s.lower().split())) /
                max(len(gap_text.split()), len(s.split())) > 0.7
                for s in seen_gaps
            )
            if not is_dup:
                seen_gaps.append(gap_text)
                clean_gaps.append({
                    "gap":          gap_text,
                    "priority":     assign_priority(gap_text),
                    "gap_type":     assign_gap_type(gap_text),
                    "evidence":     clean_text(g.get("evidence", "")),
                    "novelty_score": compute_novelty_score(gap_text, seen_gaps[:-1]),
                })
        data["gaps"] = sorted(
            clean_gaps,
            key=lambda x: (
                {"High":0,"Medium":1,"Low":2}.get(x["priority"],1),
                -x["novelty_score"]
            )
        )

    # Deduplicate limitations list
    if "limitations_list" in data:
        data["limitations_list"] = remove_duplicates(data["limitations_list"])

    # Deduplicate future directions list
    if "future_list" in data:
        data["future_list"] = remove_duplicates(data["future_list"])

    return data


# ─────────────────────────────────────────────
# TEXT PROCESSING MODULE (from paper)
# ─────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages  = [p.extract_text() for p in reader.pages if p.extract_text()]
        text   = "\n\n".join(pages)
        text   = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # fix hyphenation
        text   = re.sub(r'\s+', ' ', text)                # normalize whitespace
        return text.strip()
    except Exception as e:
        print(f"PDF error: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 400) -> list:
    """Smart sentence-boundary chunking."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur, cur_len = [], "", 0
    for s in sentences:
        wc = len(s.split())
        if cur_len + wc > chunk_size:
            if cur.strip(): chunks.append(cur.strip())
            cur, cur_len = s, wc
        else:
            cur += " " + s; cur_len += wc
    if cur.strip(): chunks.append(cur.strip())
    return chunks


def extract_section(text: str, keywords: list) -> str:
    """Extract a named section from LLM output."""
    for kw in keywords:
        m = re.search(
            rf'{re.escape(kw)}[:\s]*(.*?)(?=\n[A-Z][A-Za-z\s]+:|$)',
            text, re.DOTALL | re.IGNORECASE
        )
        if m: return m.group(1).strip()[:1500]
    return ""


def extract_list_items(text: str, section_keywords: list) -> list:
    """Extract bullet/numbered list items from a section."""
    section = extract_section(text, section_keywords)
    if not section: return []
    items = []
    for line in section.splitlines():
        line = line.strip()
        cleaned = re.sub(r'^[\d\.\-\*\•]+\s*', '', line).strip()
        if len(cleaned) > 20:
            items.append(clean_text(cleaned))
    return items


# ─────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────
def store_in_rag(papers: list) -> int:
    stored = 0
    for i, paper in enumerate(papers):
        text = f"{paper.get('title','')}. {paper.get('abstract','')}"
        for j, chunk in enumerate(chunk_text(text)):
            if len(chunk.split()) < 10: continue
            doc_id = f"{paper['title'][:40].replace(' ','_')}_{i}_{j}"
            try:
                collection.add(
                    documents  = [chunk],
                    embeddings = [embed_model.encode(chunk).tolist()],
                    ids        = [doc_id],
                    metadatas  = [{
                        "title":  paper.get("title",     ""),
                        "source": paper.get("source",    ""),
                        "year":   str(paper.get("published", "")),
                    }]
                )
                stored += 1
            except Exception:
                pass  # skip duplicate IDs
    print(f"  RAG: {stored} new chunks stored.")
    return stored


def retrieve_context(query: str, top_k: int = 5) -> str:
    results = collection.query(
        query_embeddings=[embed_model.encode(query).tolist()],
        n_results=top_k
    )
    docs = results.get("documents", [[]])
    return "\n\n---\n\n".join(docs[0]) if docs and docs[0] else ""


# ─────────────────────────────────────────────
# LITERATURE SEARCH
# ─────────────────────────────────────────────
def search_arxiv(query: str, max_results: int = 5) -> list:
    try:
        url  = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{urllib.parse.quote(query)}"
            f"&start=0&max_results={max_results}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        feed = feedparser.parse(url)
        return [{
            "title":     e.title.replace("\n", " ").strip(),
            "abstract":  e.summary.replace("\n", " ").strip(),
            "published": e.get("published", ""),
            "url":       e.get("link", ""),
            "source":    "arXiv",
        } for e in feed.entries]
    except Exception as e:
        print(f"ArXiv error: {e}"); return []


def search_semantic_scholar(query: str, limit: int = 5) -> list:
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": limit,
                    "fields": "title,abstract,year,citationCount,url"},
            timeout=30
        )
        if r.status_code != 200: return []
        return [{
            "title":     p.get("title",         ""),
            "abstract":  p.get("abstract",      "") or "",
            "published": str(p.get("year",      "")),
            "citations": p.get("citationCount", 0),
            "url":       p.get("url",           ""),
            "source":    "Semantic Scholar",
        } for p in r.json().get("data", [])]
    except Exception as e:
        print(f"Semantic Scholar error: {e}"); return []


def get_latest_research(topic: str) -> list:
    combined     = search_arxiv(topic, 5) + search_semantic_scholar(topic, 5)
    seen, unique = set(), []
    for p in combined:
        key = p["title"].lower().strip()
        if key and key not in seen:
            seen.add(key); unique.append(p)
    print(f"  Search: {len(unique)} unique papers for '{topic}'")
    return unique


# ─────────────────────────────────────────────
# OLLAMA CALLS
# ─────────────────────────────────────────────
def call_ollama(model: str, prompt: str) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt,
                  "system": SYSTEM_PROMPT, "stream": False},
            timeout=180
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"ERROR: {e}"


# ─────────────────────────────────────────────
# PECO EXTRACTION (from paper architecture)
# ─────────────────────────────────────────────
def extract_peco(text: str) -> dict:
    """Extract Population, Exposure, Comparison, Outcome."""
    prompt = f"""Extract PECO from this research content.
Respond EXACTLY in this format:
P: [who/what was studied]
E: [intervention/method/approach used]
C: [comparison/baseline]
O: [main outcome/finding]

Content: {text[:2000]}"""
    raw = call_ollama(MISTRAL_MODEL, prompt)
    peco = {}
    for field in ["P", "E", "C", "O"]:
        m = re.search(rf'^{field}:\s*(.+)', raw, re.MULTILINE)
        peco[field] = m.group(1).strip() if m else "Not identified"
    return peco


# ─────────────────────────────────────────────
# BRAIN — HYBRID ANALYSIS PIPELINE (from paper)
# ─────────────────────────────────────────────
def build_analysis_response(title: str, content: str) -> dict:
    """
    Complete pipeline matching paper architecture:
    Input → PECO → Mistral (analysis) → Llama3 (refinement) → Validate → Output
    """
    text = content[:3500]

    # ── PECO Extraction ──────────────────────
    print(f"  PECO: extracting framework...")
    peco = extract_peco(text)

    # ── Step 1: Mistral — Deep Analysis ──────
    print(f"  Step 1: Mistral deep analysis...")
    mistral_out = call_ollama(MISTRAL_MODEL, f"""Analyze this research content carefully.

You MUST provide each section EXACTLY ONCE. Do NOT repeat any section.

SUMMARY:
[3-5 sentences on main objectives, methods, and findings]

LIMITATIONS:
[bullet list — each limitation on new line starting with -]
[include: methodological, data, scope, technical, generalizability issues]

RESEARCH GAPS:
[numbered list — each gap on new line starting with 1. 2. 3. etc]
[be SPECIFIC — mention exact unexplored areas, not generic statements]
[each gap must be DIFFERENT from the others]

FUTURE WORK:
[bullet list — each direction on new line starting with -]
[must be DIFFERENT from the gaps listed above]

Research Content:
{text}""")

    # ── Step 2: Llama3 — Refinement ──────────
    print(f"  Step 2: Llama3 refinement...")
    llama_out = call_ollama(LLAMA_MODEL, f"""Convert this analysis into a clean academic report.

Rules:
- Each section appears EXACTLY ONCE
- No repeated content between sections
- Gaps must be DIFFERENT from Future Work
- Use formal academic language

Sections needed:
SUMMARY:
LIMITATIONS:
RESEARCH GAPS:
FUTURE WORK:

Raw Analysis:
{mistral_out}""")

    # ── Step 3: Parse all sections ───────────
    summary          = extract_section(llama_out, ["SUMMARY", "Summary"])
    limitations_text = extract_section(llama_out, ["LIMITATIONS", "Limitations", "Key Limitations"])
    future_text      = extract_section(llama_out, ["FUTURE WORK", "Future Work", "FUTURE"])

    limitations_list = extract_list_items(llama_out, ["LIMITATIONS", "Limitations"])
    future_list      = extract_list_items(llama_out, ["FUTURE WORK", "Future Work"])

    # ── Step 4: Parse gaps ───────────────────
    raw_gaps, in_gap = [], False
    for line in [l.strip() for l in mistral_out.splitlines() if l.strip()]:
        if re.match(r'^RESEARCH GAP', line, re.I): in_gap = True; continue
        if re.match(r'^(FUTURE|LIMITATION|SUMMARY)', line, re.I): in_gap = False
        if in_gap:
            cleaned = re.sub(r'^[\d\.\-\*\•]+\s*', '', line).strip()
            cleaned = clean_text(cleaned)
            if len(cleaned) > 20:
                raw_gaps.append(cleaned)

    # Fallback
    if not raw_gaps:
        fallback_section = extract_list_items(llama_out, ["RESEARCH GAPS", "Research Gaps"])
        raw_gaps = fallback_section[:8]

    # Build gap objects (validation happens in validate_output)
    gaps = [{"gap": g, "evidence": f"Identified via Mistral analysis of: {title}"}
            for g in raw_gaps[:10]]

    # ── Step 5: Store in memory ───────────────
    memory["literature"].append({
        "title":    title,
        "year":     "2024",
        "abstract": content[:300],
    })
    memory["evidence"].append({
        "paper":      title,
        "findings":   summary[:400] if summary else "",
        "limitation": limitations_text[:400] if limitations_text else "",
    })

    # ── Step 6: Build + Validate response ────
    result = {
        "title":             title,
        "peco":              peco,
        "summary":           summary or mistral_out[:400],
        "limitations":       limitations_text,
        "limitations_list":  limitations_list,
        "gaps":              gaps,
        "future":            future_text,
        "future_list":       future_list,
        "model_used":        f"{MISTRAL_MODEL} → {LLAMA_MODEL}",
        "word_count":        len(content.split()),
    }

    result = validate_output(result)

    # Store validated gaps in memory Gap Table
    memory["gaps"].extend([{
        "gap":      g["gap"],
        "evidence": g.get("evidence",""),
        "priority": g.get("priority","Medium"),
    } for g in result["gaps"]])

    # Add gap clusters
    result["gap_clusters"] = cluster_gaps(result["gaps"])

    # Add stats
    result["stats"] = {
        "total_gaps":    len(result["gaps"]),
        "high_priority": sum(1 for g in result["gaps"] if g["priority"]=="High"),
        "med_priority":  sum(1 for g in result["gaps"] if g["priority"]=="Medium"),
        "low_priority":  sum(1 for g in result["gaps"] if g["priority"]=="Low"),
        "avg_novelty":   int(sum(g.get("novelty_score",50) for g in result["gaps"]) /
                         max(len(result["gaps"]),1)),
    }

    print(f"  ✅ Done — {result['stats']['total_gaps']} gaps, "
          f"avg novelty: {result['stats']['avg_novelty']}/100")
    return result


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/api/ping")
def ping():
    return jsonify({
        "status":   "ok",
        "version":  "A+",
        "models":   f"{MISTRAL_MODEL} + {LLAMA_MODEL}",
        "rag":      "ChromaDB + SentenceTransformer",
        "pipeline": "PECO → Mistral → Llama3 → Validate",
        "papers":   len(memory["literature"]),
        "gaps":     len(memory["gaps"]),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    msg = request.get_json().get("message", "").strip()
    if not msg: return jsonify({"error": "Empty message"}), 400
    ctx = retrieve_context(msg, top_k=3)
    prompt = (f"Use this context from analyzed papers:\n{ctx}\n\nQuestion: {msg}"
              if ctx else msg)
    return jsonify({"reply": call_ollama(LLAMA_MODEL, prompt)})


@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    data  = request.get_json()
    text  = data.get("text", "").strip()
    title = data.get("title", "Pasted Text")
    if not text or len(text) < 50:
        return jsonify({"error": "Provide at least 50 characters."}), 400
    store_in_rag([{"title": title, "abstract": text[:1000], "source": "User"}])
    return jsonify(build_analysis_response(title, text))


@app.route("/api/analyze/pdf", methods=["POST"])
def analyze_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files supported"}), 400
    text = extract_text_from_pdf(file.read())
    if not text or len(text) < 50:
        return jsonify({"error": "Could not extract text. Scanned PDF?"}), 400
    title = file.filename.replace(".pdf", "").replace("_", " ")
    store_in_rag([{"title": title, "abstract": text[:2000], "source": "PDF"}])
    return jsonify(build_analysis_response(title, text))


@app.route("/api/analyze/topic", methods=["POST"])
def analyze_topic():
    topic = request.get_json().get("topic", "").strip()
    if not topic: return jsonify({"error": "No topic provided"}), 400

    papers = get_latest_research(topic)
    if not papers:
        return jsonify({"error": "No papers found. Try different keywords."}), 404

    store_in_rag(papers)
    rag_context = retrieve_context(topic, top_k=6)

    papers_text = "".join(
        f"\nTitle: {p['title']}\nSource: {p['source']}\n"
        f"Abstract: {p.get('abstract','')[:400]}\n{'─'*40}\n"
        for p in papers[:8]
    )
    combined = f"RAG Context:\n{rag_context}\n\nPapers:\n{papers_text}"

    result = build_analysis_response(f"Topic: {topic}", combined)
    result["papers_found"] = len(papers)
    result["papers"] = [{
        "title":     p["title"],
        "source":    p["source"],
        "published": p.get("published",""),
        "url":       p.get("url",""),
        "citations": p.get("citations", 0),
    } for p in papers]
    return jsonify(result)


@app.route("/api/report")
def report():
    if not memory["literature"]:
        return jsonify({"error": "No analyses done yet."}), 400

    high   = [g for g in memory["gaps"] if g.get("priority")=="High"]
    medium = [g for g in memory["gaps"] if g.get("priority")=="Medium"]
    low    = [g for g in memory["gaps"] if g.get("priority")=="Low"]

    lines = [
        "# GapAgent Research Gap Report",
        f"*Generated by GapAgent A+ | Pipeline: PECO → Mistral → Llama3 → Validate*",
        "",
        f"## Session Statistics",
        f"- Papers/Topics Analyzed: **{len(memory['literature'])}**",
        f"- Total Gaps Identified: **{len(memory['gaps'])}**",
        f"- High Priority: **{len(high)}** | Medium: **{len(medium)}** | Low: **{len(low)}**",
        "",
        "## Analyzed Literature",
    ]
    for i, item in enumerate(memory["literature"], 1):
        t = item["title"] if isinstance(item, dict) else item
        lines.append(f"{i}. {t}")

    lines += ["", "## Evidence Summary"]
    for ev in memory["evidence"]:
        if isinstance(ev, dict) and ev.get("findings"):
            lines.append(f"**{ev['paper']}**")
            lines.append(f"- Findings: {ev['findings'][:200]}")
            lines.append(f"- Limitations: {ev['limitation'][:200]}")
            lines.append("")

    lines += ["## Research Gaps by Priority", ""]

    if high:
        lines.append("### 🔴 High Priority Gaps")
        for g in high:
            lines.append(f"- **{g['gap']}**")
            if g.get("evidence"): lines.append(f"  - *Evidence: {g['evidence'][:100]}*")
        lines.append("")

    if medium:
        lines.append("### 🟡 Medium Priority Gaps")
        for g in medium:
            lines.append(f"- {g['gap']}")
        lines.append("")

    if low:
        lines.append("### 🟢 Low Priority Gaps")
        for g in low:
            lines.append(f"- {g['gap']}")
        lines.append("")

    lines += [
        "## System Notes",
        f"- Models: {MISTRAL_MODEL} (analysis) + {LLAMA_MODEL} (refinement)",
        f"- RAG: ChromaDB + all-MiniLM-L6-v2 embeddings",
        f"- Post-processing: deduplication + priority scoring + novelty index",
        f"- Architecture: Input → PECO → Brain → Memory → Output",
    ]
    return jsonify({
        "report": "\n".join(lines),
        "stats": {
            "papers": len(memory["literature"]),
            "total":  len(memory["gaps"]),
            "high":   len(high),
            "medium": len(medium),
            "low":    len(low),
        }
    })


@app.route("/api/memory")
def get_memory():
    return jsonify({
        **memory,
        "stats": {
            "papers": len(memory["literature"]),
            "gaps":   len(memory["gaps"]),
            "high":   sum(1 for g in memory["gaps"] if g.get("priority")=="High"),
            "medium": sum(1 for g in memory["gaps"] if g.get("priority")=="Medium"),
            "low":    sum(1 for g in memory["gaps"] if g.get("priority")=="Low"),
        }
    })


@app.route("/api/memory/clear", methods=["POST"])
def clear_memory():
    for k in memory: memory[k].clear()
    return jsonify({"status": "cleared"})


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*58)
    print("  🔬 GapAgent — A+ Final Version")
    print("="*58)
    print(f"  Pipeline : Input → PECO → Mistral → Llama3 → Validate")
    print(f"  Models   : {MISTRAL_MODEL} (analysis) + {LLAMA_MODEL} (refine)")
    print(f"  RAG      : ChromaDB + SentenceTransformer")
    print(f"  Search   : ArXiv + Semantic Scholar")
    print(f"  Fixes    : Dedup + Priority + Novelty + CleanText")
    print(f"  URL      : http://localhost:8000")
    print("="*58 + "\n")
    app.run(debug=True, port=8000, threaded=True)