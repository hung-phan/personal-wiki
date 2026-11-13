---
name: retrieval-augmented-generation
description: Retrieval-augmented generation (RAG) explained with runnable local code — why RAG (paste relevant knowledge into the prompt), a brittle keyword toy, embeddings and encoders (all-MiniLM-L6-v2, text-embedding-3, cosine similarity, typos and synonyms), chunking with size and overlap and the lost-context problem, vector stores (Chroma, FAISS, pgvector) vs encoder choice, LangChain abstractions vs a native pipeline, history-aware query rewriting, evaluation with a golden set (MRR, nDCG, recall@k, keyword coverage, LLM-as-judge), BM25 keyword search, hybrid search with reciprocal rank fusion, cross-encoder and LLM reranking, contextual chunk headers, query expansion, hierarchical, graph and agentic RAG. Study guide for the LLM Engineering course, Week 5.
created: 2026-11-13
updated: 2026-11-13
tags: [llm, llm-engineering, rag, retrieval, embeddings, vector-database, chunking, reranking, evaluation, mrr, ndcg, study-guide, llm-engineering-course]
---
# Retrieval-Augmented Generation (RAG)

> **How to read this note.** Study guide for Week 5 of *AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents* (lectures 106–137). The course builds a question-answering assistant over the shared drive of a fictional insurance-tech company (76 markdown files: employees, products, contracts), first with LangChain and Chroma, then natively with an LLM-based chunker, query rewriting and reranking, measuring every change. Here a smaller fictional company ("Lumora", 10 documents) stands in, and everything runs locally with the course's own encoder, **all-MiniLM-L6-v2**, plus a cross-encoder reranker (each ~90 MB, CPU is fine). Install: `VIRTUAL_ENV=/Users/phahng/Desktop/workarea/.venv uv pip install sentence-transformers`.

## Why This Exists

**Problem.** An LLM knows nothing about your company, and its knowledge stops at a cutoff. Pasting all your documents into every prompt is slow, costly and eventually impossible. RAG fixes that, but it's a pile of tunable parts — chunk size, encoder, k, reranking, prompt — where every choice can silently break answers.

**Key insight.** RAG is a **lookup before the LLM call**: turn the question into a search over your documents, paste the few most relevant pieces into the prompt, and let the model answer from them. Embeddings make the search match *meaning* instead of exact words. Because RAG is empirical, the most valuable thing you build is the **evaluation set** that tells you whether a change helped.

**Reach for this when:**
- An assistant must answer from private, recent or large document collections.
- Answers need sources you can show.
- A RAG system "sometimes" gives wrong answers and you need to find out why.

**Don't reach for this when:**
- The model must learn a *skill or style*, not facts — that's fine-tuning ([qlora-fine-tuning](qlora-fine-tuning.md)).
- All the knowledge fits comfortably in the context window and rarely changes — just include it (with prompt caching).
- You need to *train* an embedding model — see [embedding-training](../ml-systems/embedding-training.md).

## Diagrams

![RAG pipeline: ingest documents, chunk, encode, store in a vector store; at query time rewrite the question, encode with the same encoder, top-k search, rerank, and paste the chunks into the prompt; evaluate retrieval with MRR, nDCG and recall@k and answers with an LLM judge](assets/rag-pipeline.svg)

## 1. The small idea: look it up first

**The question.** How do you make a model answer "What's our CEO's background?" when it has never seen your company?

Find the relevant document and put it in the prompt. The course's first version keeps documents in a dictionary keyed by surname or product name and includes any document whose key appears in the question:

```python
import re
docs = {"lancaster": "Avery Lancaster is the CEO. She founded the company in 2015.",
        "chen": "Alex Chen runs marketing. He reports to Avery Lancaster."}
def retrieve(question):                                    # toy RAG: a doc is relevant if its KEY word appears
    words = set(re.sub(r"[^\w\s]", "", question.lower()).split())
    return [text for key, text in docs.items() if key in words]
for q in ("What does Lancaster do?", "Who is Avery?", "Who is the chief executive?"):
    print(f"{q:30s} -> {retrieve(q)}")
# Expected output:
# What does Lancaster do?        -> ['Avery Lancaster is the CEO. She founded the company in 2015.']
# Who is Avery?                  -> []
# Who is the chief executive?    -> []
```

It works for the exact surname and fails for a first name or a synonym. Real questions don't use your filenames. A further gotcha from the course: in a chat, conversation history can make the model answer correctly even when retrieval failed — test retrieval in fresh sessions.

## 2. The big idea: search by meaning with embeddings

**The question.** How can "chief executive" find a document that says "CEO"?

Use an **encoder** (embedding model): a network that maps a whole text to one vector so that texts with similar meaning land close together. Unlike a chat LLM, which predicts the next token, an encoder produces a single output for the whole input — the same family as BERT. Examples: all-MiniLM-L6-v2 (384 numbers per text, free, local), OpenAI's text-embedding-3-small/-large (1,536 / 3,072), Gemini embeddings. Similarity is usually the **cosine** of the angle between vectors. (The famous word2vec arithmetic — king − man + woman ≈ queen — is the same idea at the word level.)

```python
from sentence_transformers import SentenceTransformer, util
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # 384-dim encoder, ~90 MB, runs on CPU
docs = ["Avery Lancaster is the CEO and founded the company.", "Alex Chen runs marketing campaigns.",
        "Our home insurance product quotes premiums in seconds.", "The cafeteria serves lunch from noon."]
D = enc.encode(docs, normalize_embeddings=True)           # one vector per document
print("vector shape:", D.shape)
for q in ("Who is the chief executive?", "Who is Avery?", "fast quotes for house cover", "whos in charge of advertsing"):
    sims = util.cos_sim(enc.encode(q, normalize_embeddings=True), D)[0]   # cosine similarity to each doc
    best = int(sims.argmax())
    print(f"{q:28s} -> {docs[best][:45]:45s} (cos {float(sims[best]):.2f})")
# Expected output:
# vector shape: (4, 384)
# Who is the chief executive?  -> Avery Lancaster is the CEO and founded the co (cos 0.46)
# Who is Avery?                -> Avery Lancaster is the CEO and founded the co (cos 0.58)
# fast quotes for house cover  -> Our home insurance product quotes premiums in (cos 0.39)
# whos in charge of advertsing -> Alex Chen runs marketing campaigns.           (cos 0.45)
```

Synonyms ("chief executive" → CEO, "house cover" → home insurance) and even misspellings ("advertsing") find the right document. The encoder and the answering LLM are separate models; the LLM only ever sees retrieved **text**, never vectors.

**The full design:**
- **Ingest** (once): split documents into chunks, encode each chunk, store text + vector + metadata in a **vector store**.
- **Query** (each question): encode the question with the *same* encoder, find the nearest chunks, paste them into the prompt with the question.

**Encoder vs vector store.** The course's strong claim: the encoder decides retrieval quality; the store (Chroma, FAISS, Pinecone, Weaviate, or vectors inside Postgres/pgvector, Mongo or Elasticsearch) is mostly an infrastructure, cost and scale decision. Swapping MiniLM for OpenAI's larger encoders gave visibly cleaner clusters in a t-SNE plot of the course's chunks and better scores. (t-SNE pictures are suggestive only; they distort global distances.) Ingest and query must use the same encoder — otherwise the dimensions don't even match.

## 3. Chunking: how big a piece to retrieve

**The question.** Why not store whole documents?

A question usually concerns part of a document, and one long document's vector averages many topics, diluting the match. So documents are split into **chunks** — the course uses LangChain's `RecursiveCharacterTextSplitter` (splits at paragraphs, then lines, then words, to a target size) with **overlap** so an answer that straddles a boundary survives. Too large: diluted matches and long prompts. Too small: chunks lose the context that makes them meaningful.

```python
def chunk(text, size, overlap):                           # fixed-size character chunks that overlap
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size]); start += size - overlap
    return out
doc = ("# Maxine Rivera\nRole: Head of Claims\n\n## Awards\nIn 2023 she won the Innovator of the Year award for "
       "launching same-day claim payments.\n\n## Career\nShe joined from a bank in 2020 and now manages 14 adjusters.")
for i, c in enumerate(chunk(doc, 90, 20)): print(i, repr(c))
hit = [c for c in chunk(doc, 90, 20) if "same-day" in c][0]   # what "who launched same-day payments?" retrieves
print("\nretrieved chunk names the person?", "Maxine" in hit)
print("fix, prefix each chunk with its title:", repr("Maxine Rivera, Head of Claims | " + hit[:45]))
# Expected output:
# 0 '# Maxine Rivera\nRole: Head of Claims\n\n## Awards\nIn 2023 she won the Innovator of the Year '
# 1 'novator of the Year award for launching same-day claim payments.\n\n## Career\nShe joined fro'
# 2 'areer\nShe joined from a bank in 2020 and now manages 14 adjusters.'
# 
# retrieved chunk names the person? False
# fix, prefix each chunk with its title: 'Maxine Rivera, Head of Claims | novator of the Year award for launching same-'
```

The chunk that answers "who launched same-day payments?" doesn't contain the person's name — it was in the header. The course hit exactly this (an award chunk without the winner's surname). Fixes: add the document title or a short summary to every chunk ("contextual" chunk headers), split on document structure (markdown headings), or have an LLM rewrite each document into self-contained chunks (the course's "pro" version: headline + summary + original text per chunk).

## 4. The pipeline in code: LangChain vs native

The course first assembles RAG from **LangChain** pieces — `DirectoryLoader`, the text splitter, `HuggingFaceEmbeddings` or `OpenAIEmbeddings`, a `Chroma` store, `vectorstore.as_retriever()` (4 chunks by default; the course uses 5), and `ChatOpenAI` — then rewrites it without LangChain using `chromadb`, LiteLLM and Pydantic. The native version is only a few lines longer, and every step is visible. LangChain's value is speed of assembly and a shared vocabulary; its cost is weight and indirection.

**Conversation history breaks naive retrieval.** A follow-up like "and what does that person earn?" has nothing to search on. Concatenating the whole history helps until the topic changes, then it drags in the old topic. The robust fix is to have an LLM **rewrite the latest question into a standalone query** before retrieving:

```python
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
docs = ["Marcus Cole, enterprise sales lead. Base salary: $145,000.",
        "Priya Natarajan, Chief Technology Officer. Base salary: $210,000.",
        "How salaries work: pay bands are reviewed every January."]
D = enc.encode(docs, normalize_embeddings=True)
def top(q): return docs[int((D @ enc.encode(q, normalize_embeddings=True)).argmax())][:40]
history = ["Tell me about Priya Natarajan.", "Priya is Lumora's CTO and leads the platform team."]
follow_up = "And what does that person earn?"
print("follow-up alone        ->", top(follow_up))                     # "that person" means nothing to the encoder
print("history + follow-up    ->", top(" ".join(history + [follow_up])))
print("rewritten standalone   ->", top("What does Priya Natarajan earn?"))   # what an LLM rewrite step produces
# Expected output:
# follow-up alone        -> Marcus Cole, enterprise sales lead. Base
# history + follow-up    -> Priya Natarajan, Chief Technology Office
# rewritten standalone   -> Priya Natarajan, Chief Technology Office
```

## 5. Evaluation: the part that matters most

**The question.** You changed chunk size from 1,000 to 500. Better or worse?

You can't tell by trying three questions. The course's three-step process:
1. **A golden set** — questions, each with keywords that a relevant chunk must contain and a reference answer, covering categories (direct facts, temporal, comparisons, numbers, relationships, questions spanning several documents, holistic "summarise the company" questions). Best drawn from real users and grown over time. The course's has 150 questions.
2. **Retrieval metrics**, computed without any LLM:
   - **MRR** (mean reciprocal rank): for each question, 1 / (rank of the first relevant chunk); average. 1.0 = always first.
   - **nDCG@k**: rewards relevant chunks near the top, discounting each by $\log_2(\text{rank}+1)$, normalised by the best possible order.
   - **recall@k**: share of questions with at least one relevant chunk in the top k. For RAG this matters more than precision: the LLM can ignore an irrelevant chunk but can't use one that was never retrieved.
   - **keyword coverage**: share of expected keywords found anywhere in the retrieved context.
3. **Answer quality** — an **LLM-as-judge** compares each answer with the reference and scores accuracy, completeness and relevance from 1 to 5 (structured output via Pydantic). Use a strong judge and keep it fixed across experiments.

Here is the same evaluation on the Lumora documents, comparing four retrievers:

```python
KB = {  # fictional insurance-tech company "Lumora": one string per document
 "employee_natarajan": "Priya Natarajan is Lumora's Chief Technology Officer. She joined in 2019 from a payments startup. She studied computer science at the University of Edinburgh and leads the platform team.",
 "employee_cole": "Marcus Cole leads enterprise sales. He was named Salesperson of the Year in 2023 after closing the Harbor Mutual deal. His base salary is $145,000.",
 "employee_okafor": "Dr. Ada Okafor heads data science. She built the pricing models behind HomeShield and previously worked at a reinsurance firm in Zurich.",
 "employee_lindqvist": "Erik Lindqvist is a support engineer in the Stockholm office. He maintains the ClaimPilot integrations and runs the on-call rota.",
 "product_homeshield": "HomeShield is Lumora's home-insurance pricing engine. Insurers send property details and receive a premium quote within two seconds.",
 "product_fleetwise": "FleetWise is a telematics product for commercial vehicle insurance. It scores driver behaviour from GPS and braking data to adjust fleet premiums monthly.",
 "product_claimpilot": "ClaimPilot triages incoming claims. It reads photos and adjuster notes, flags likely fraud, and routes simple claims for same-day payment.",
 "contract_harbor": "Harbor Mutual signed a three-year contract for HomeShield and ClaimPilot in March 2023, worth $2.4 million per year, with a 99.9% uptime guarantee.",
 "contract_greenline": "Greenline Logistics uses FleetWise for 1,200 trucks under a two-year contract starting 2024. Pricing is per vehicle per month.",
 "company_about": "Lumora was founded in 2017 in Amsterdam. It has 240 employees across Amsterdam, Stockholm and Austin and sells software to insurers, not policies to consumers.",
}
GOLDEN = [  # (question, keyword that must appear in a relevant document)
 ("Who went to the University of Edinburgh?", "Edinburgh"),
 ("Which product decides how much to charge for house insurance?", "pricing engine"),
 ("Who was the top salesperson in 2023?", "Salesperson of the Year"),
 ("What does the fraud detection product do?", "flags likely fraud"),
 ("How much is the Harbor Mutual deal worth?", "$2.4 million"),
 ("Which customer has 1,200 trucks?", "1,200 trucks"),
 ("Where was the company founded?", "Amsterdam"),
 ("Who looks after on-call duty?", "on-call"),
]
import re, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
ids, texts = list(KB), list(KB.values())
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
D = enc.encode(texts, normalize_embeddings=True)
toks = lambda s: re.findall(r"\w+", s.lower())
docs_t = [toks(t) for t in texts]; avgdl = np.mean([len(d) for d in docs_t])
def bm25(q, k1=1.5, b=0.75):                               # classic keyword scoring
    s = np.zeros(len(texts))
    for w in set(toks(q)):
        df = sum(w in d for d in docs_t); idf = np.log(1 + (len(texts) - df + 0.5) / (df + 0.5))
        for i, d in enumerate(docs_t):
            f = d.count(w); s[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
    return list(np.argsort(-s))
dense = lambda q: list(np.argsort(-(D @ enc.encode(q, normalize_embeddings=True))))
def hybrid(q, k=60):                                       # reciprocal rank fusion of both rankings
    score = {}
    for ranking in (dense(q), bm25(q)):
        for r, i in enumerate(ranking): score[i] = score.get(i, 0) + 1 / (k + r + 1)
    return sorted(score, key=lambda i: -score[i])
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")   # reads (query, doc) TOGETHER: slower, sharper
def rerank(q, n=5):
    top = dense(q)[:n]; s = ce.predict([(q, texts[i]) for i in top])
    return [top[i] for i in np.argsort(-s)] + dense(q)[n:]
def metrics(rank_fn):
    rr, ndcg, r1, r3 = [], [], [], []
    for q, kw in GOLDEN:
        rel = [kw in texts[i] for i in rank_fn(q)]          # relevance judged by the golden keyword
        first = rel.index(True)                            # 0-based rank of the first relevant doc
        rr.append(1 / (first + 1)); r1.append(first < 1); r3.append(first < 3)
        ndcg.append(sum(r / np.log2(i + 2) for i, r in enumerate(rel[:5])) / 1.0)   # one relevant doc: ideal DCG = 1
    return np.mean(rr), np.mean(ndcg), np.mean(r1), np.mean(r3)
for name, fn in (("keyword BM25", bm25), ("dense MiniLM", dense), ("hybrid RRF", hybrid), ("dense+rerank", rerank)):
    print("%-13s MRR %.3f  nDCG@5 %.3f  recall@1 %.2f  recall@3 %.2f" % ((name,) + metrics(fn)))
# Expected output:
# keyword BM25  MRR 0.854  nDCG@5 0.891  recall@1 0.75  recall@3 1.00
# dense MiniLM  MRR 0.938  nDCG@5 0.954  recall@1 0.88  recall@3 1.00
# hybrid RRF    MRR 1.000  nDCG@5 1.000  recall@1 1.00  recall@3 1.00
# dense+rerank  MRR 1.000  nDCG@5 1.000  recall@1 1.00  recall@3 1.00
```

- **BM25** is classic keyword search (term frequency, inverse document frequency, length normalisation). It wins on exact names and numbers, loses on paraphrases.
- **Dense** (embeddings) wins on paraphrases, can miss exact rare terms.
- **Hybrid** search fuses the two rankings with **reciprocal rank fusion** (each document scores $\sum 1/(60 + \text{rank})$ across rankers) and fixes both kinds of miss.
- **Reranking** retrieves a generous top-k with the fast encoder, then rescores each (question, chunk) pair with a **cross-encoder** that reads them together — slower per pair, much sharper.

Eight questions over ten documents is far too small to rank methods seriously; it shows the mechanics. The course's ladder on 150 questions, MRR at each step: baseline MiniLM, 1,000-character chunks, k=5 → **0.73**; 500-character chunks, k=10 → 0.76; text-embedding-3-small → 0.78; text-embedding-3-large → 0.79 (answer accuracy 3.99 → 4.21 out of 5); the "pro" pipeline (LLM-rewritten chunks, query rewriting plus expansion, LLM reranking, a stronger answer model) → **0.91**, accuracy 4.62. The last step changed several things at once, so its gain can't be attributed to any one of them.

## 6. Ten ways to improve RAG

The course's list, with what each targets:
1. **Chunking R&D** — size, overlap, structure-aware splitting. Measure.
2. **Encoder R&D** — try several; for images, caption them and embed the text; convert PDFs to clean markdown with ordinary libraries first.
3. **Prompting** — fixed instructions first (cacheable), the current date, how to use history, "say you don't know".
4. **Document pre-processing** — have an LLM rewrite documents into self-contained chunks with headings and summaries.
5. **Query rewriting** — turn the latest message into a standalone search query.
6. **Query expansion** — search with several phrasings, merge and de-duplicate results.
7. **Reranking** — retrieve wide (say 20), rerank, keep the best 5–10. The course reranks with an LLM; dedicated cross-encoder rerankers are cheaper and deterministic.
8. **Hierarchical RAG** — add summary documents at several levels so "holistic" questions find an overview.
9. **GraphRAG** — link related chunks (by metadata or in a graph database) so relationships can be followed.
10. **Agentic RAG** — let an agent decide when and how to search, with several tools (vector search, keyword search, file reading). Still RAG, just adaptive.

Not in the course but standard: **hybrid search** (§5) and **metadata filtering** (search only contracts, or only 2024 documents).

**"Is RAG dead?"** Long context windows and agents with search tools are sometimes said to replace it. The course's answer: an agent that searches is doing retrieval; and putting everything in context is slow, costly and degrades with length ("lost in the middle"). The techniques above remain the toolbox.

## Trade-offs

- **Chunk size.** Precise matches vs complete context; tune on the golden set.
- **k (chunks per question).** Higher recall vs longer, noisier, costlier prompts.
- **Local vs hosted encoder.** Free and private vs (often) higher quality and a per-token cost.
- **Dense vs keyword vs hybrid.** Meaning vs exact terms vs both, at the cost of two indexes.
- **Reranking.** Better ordering vs extra latency (cross-encoder) or cost (LLM).
- **LangChain vs native code.** Speed of assembly vs transparency and fewer dependencies.
- **LLM-rewritten chunks.** Much better retrieval vs ingest cost and the risk of the rewrite dropping facts.

## Common Pitfalls

- **No golden set.** Without it, every change is a guess. Build it first, even 30 questions.
- **Changing several things at once.** You learn nothing about which helped.
- **Different encoders at ingest and query.** Wrong or impossible similarities. Re-index when you switch.
- **Chunks without context.** A chunk that says "she won the award" can't answer "who won?". Add titles or summaries.
- **Retrieving on the raw follow-up message.** Rewrite it into a standalone query.
- **Concatenating all history for retrieval.** Pollutes search after a topic change.
- **Grading with keyword matches only.** Fast but crude; check answers with a judge and by hand.
- **Teaching to the test.** Telling the answer model the judge's exact criteria inflates scores.
- **Trusting t-SNE pictures.** Use metrics to compare encoders.
- **Feeding untrusted documents to a tool-using model.** Retrieved text can carry prompt-injection instructions.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| Answers from private or fresh documents | RAG | Fine-tuning on the documents | Facts change; retrieval updates instantly |
| Questions with exact names, IDs, numbers | Hybrid (BM25 + dense) | Dense alone | Keywords catch rare exact terms |
| Paraphrased, fuzzy questions | Dense embeddings | Keyword search | Matches meaning |
| Right chunk retrieved but ranked low | Retrieve wide + cross-encoder rerank | Raising k only | Better order, short prompt |
| Follow-up questions in chat | LLM query rewriting | Raw message or full history | Standalone, on-topic query |
| "Summarise everything about X" | Hierarchical summaries / agentic search | Top-5 chunks | Spanning questions need overviews |
| Choosing chunk size or encoder | Golden-set MRR/recall, one change at a time | Eyeballing answers | Measurable, attributable |

## References

- [Course repository: ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering)
- [Lewis et al. — Retrieval-Augmented Generation for Knowledge-Intensive NLP (2020)](https://arxiv.org/abs/2005.11401) — the original RAG paper.
- [Sentence-Transformers (SBERT)](https://sbert.net/) · [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [ms-marco-MiniLM-L-6-v2 cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) · [MTEB embedding leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Chroma](https://docs.trychroma.com/) · [FAISS](https://github.com/facebookresearch/faiss) · [pgvector](https://github.com/pgvector/pgvector) · [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Anthropic — Contextual Retrieval (2024)](https://www.anthropic.com/news/contextual-retrieval) — chunk-level context, hybrid search and reranking measured.
- [Cormack, Clarke & Büttcher — Reciprocal Rank Fusion (SIGIR 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Edge et al. — GraphRAG (2024)](https://arxiv.org/abs/2404.16130) · [Liu et al. — Lost in the Middle (2023)](https://arxiv.org/abs/2307.03172)
- [Mikolov et al. — word2vec (2013)](https://arxiv.org/abs/1301.3781) · [van der Maaten & Hinton — t-SNE (JMLR 2008)](https://www.jmlr.org/papers/v9/vandermaaten08a.html)

## See Also

- [embedding-training](../ml-systems/embedding-training.md) — how encoders are trained (contrastive learning) and why reconstruction ≠ similarity.
- [frontier-apis-and-tools](frontier-apis-and-tools.md) — context-on-demand in a chat UI, and the tool loop that agentic RAG uses.
- [agentic-ai-systems](agentic-ai-systems.md) — Week 8 reuses RAG over 800k products as one agent's tool.
- [choosing-llms](choosing-llms.md) — model vs business metrics; error bars on small evaluations.
