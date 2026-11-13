"""Diagrams for wiki/machine-learning/llm-engineering notes.  Run: python3 scripts/llm_engineering_diagrams.py wiki/machine-learning/llm-engineering/assets [foundations ...]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_svg as ps
from paper_svg import T, R, A, P, C, header, matrix, colvec, note
ps.OUT = sys.argv[1]
os.makedirs(ps.OUT, exist_ok=True)
svg = ps.svg
WHICH = sys.argv[2:] or ["all"]
def want(n): return "all" in WHICH or n in WHICH

# ---------------- Week 1: anatomy of one LLM call ----------------
if want("foundations"):
    b = [header("LLM FOUNDATIONS", "One API call: everything goes in, one token comes out at a time", 960)]
    b.append(T(40, 104, "1. the request: a list of messages", "h2"))
    msgs = [("system", "You are a snarky assistant.", "yel"), ("user", "What is 2+2?", "blu"),
            ("assistant", "4, obviously.", "grn"), ("user", "And 3+3?", "blu")]
    for k, (role, text, c) in enumerate(msgs):
        y = 122 + k * 44
        b.append(R(40, y, 260, 36, c, 4)); b.append(T(50, y + 15, role, "numb")); b.append(T(50, y + 30, text, "num"))
    b.append(T(170, 312, "the model never remembers: your code", "hand", "middle"))
    b.append(T(170, 330, "re-sends the whole history every call", "hand", "middle"))
    b.append(A(310, 210, 360, 210)); b.append(T(335, 200, "chat template", "mut", "middle"))
    b.append(T(370, 104, "2. one flat token sequence", "h2"))
    toks = ["<sys>", "You", " are", " a", " snark", "y", "…", "<user>", "What", " is", " 2", "+", "2", "?", "…", "<asst>"]
    for k, t in enumerate(toks):
        x = 370 + (k % 8) * 62; y = 150 + (k // 8) * 44
        b.append(R(x, y, 56, 30, "pur" if t.startswith("<") else "cell", 3)); b.append(T(x + 28, y + 20, t, "num", "middle"))
    b.append(T(620, 260, "special tokens mark who is speaking; the request", "mut", "middle"))
    b.append(T(620, 276, "ends with an open assistant turn to be continued", "mut", "middle"))
    b.append(R(420, 300, 400, 60, "pnk", 8)); b.append(T(620, 326, "LLM: predict the next token", "h2", "middle"))
    b.append(T(620, 346, "probabilities over ~100k–200k possible tokens", "mut", "middle"))
    b.append(A(620, 290, 620, 300))
    b.append(P("M820,330 C 900,330 900,200 870,180", "lnr", "arr"))
    b.append(T(905, 260, "append the", "hand", "middle")); b.append(T(905, 276, "chosen token,", "hand", "middle")); b.append(T(905, 292, "repeat", "hand", "middle"))
    b.append(T(620, 386, "streamed back token by token until an end-of-turn token or max_tokens", "num", "middle"))
    # context window bar
    b.append(T(40, 424, "3. the context window holds the request AND the reply", "h2"))
    parts = [("system", 60, "yel"), ("user 1", 40, "blu"), ("assistant 1", 150, "grn"), ("user 2", 40, "blu"), ("assistant 2", 150, "grn"),
             ("…", 120, "cell"), ("new user", 40, "blu"), ("reply being generated", 170, "pnk")]
    x = 40
    for name, w, c in parts:
        ww = w * 1.15; b.append(R(x, 438, ww, 34, c, 2)); b.append(T(x + ww / 2, 459, name, "num", "middle")); x += ww
    b.append(T(x + 6, 459, "| limit", "numb"))
    b.append(note(40, 488, 430, 76, "You pay for", ["every input token on every call (history included)", "+ every output token, including hidden reasoning tokens"], "handk"))
    b.append(note(490, 488, 440, 76, "Rules of thumb (English text)", ["1 token ≈ 4 characters ≈ 0.75 words", "code, numbers and rare words cost more tokens"], "handk"))
    svg("llm-call.svg", 960, 580, "".join(b))

# ---------------- Week 1: model types and ways to run them ----------------
if want("foundations"):
    b = [header("LLM FOUNDATIONS", "Three kinds of model, three ways to use one", 960)]
    cols = [("base", "pretrained on web text", "continues any text:", "autocomplete", "fine-tuning a new skill", "yel"),
            ("chat / instruct", "+ trained on conversations", "+ human feedback (RLHF):", "follows roles & instructions", "most apps; fast, fluent", "blu"),
            ("reasoning", "+ trained to think step by step", "before answering", "(hidden 'thinking' tokens)", "puzzles, maths, code, planning", "pnk")]
    for k, (name, l1, l2, l3, use, c) in enumerate(cols):
        x = 40 + k * 300
        b.append(R(x, 100, 270, 150, c, 8)); b.append(T(x + 135, 126, name, "h2", "middle"))
        for i, l in enumerate((l1, l2, l3)): b.append(T(x + 135, 154 + i * 20, l, "ink", "middle"))
        b.append(T(x + 135, 236, "use for: " + use, "hand", "middle"))
        if k < 2: b.append(A(x + 272, 175, x + 298, 175))
    b.append(T(480, 272, "hybrid models decide per question how long to think; more thinking = more tokens, cost and latency", "mut", "middle"))
    b.append(T(40, 310, "ways to run a model", "h2"))
    ways = [("chat product", ["ChatGPT, Claude, Gemini apps", "UI, memory, web search are", "product features, not the model"]),
            ("cloud API", ["direct (OpenAI, Anthropic, Google…)", "managed (Bedrock, Vertex, Azure)", "routers (OpenRouter), fast hosts (Groq)"]),
            ("local", ["Ollama: optimised C++ + GGUF weights", "HF transformers: the raw Python model,", "the route to fine-tuning"])]
    for k, (name, lines) in enumerate(ways):
        b.append(note(40 + k * 300, 324, 280, 120, name, lines, "handk"))
    b.append(note(40, 460, 880, 90, "One client for all of them",
                  ["Almost every provider and Ollama expose an OpenAI-compatible endpoint, so the same OpenAI client works:",
                   "change base_url and api_key, keep the code. Open-weight ≠ open source: the weights are public, the training data usually isn't."]))
    svg("llm-landscape.svg", 960, 570, "".join(b))

# ---------------- Week 2: tool calling ----------------
if want("tools"):
    b = [header("FRONTIER APIs & TOOLS", "Tool calling: the model only asks, your code does", 960)]
    b.append(T(40, 104, "how it feels", "h2"))
    b.append(R(40, 120, 120, 44, "pnk", 6)); b.append(T(100, 147, "LLM", "h2", "middle"))
    b.append(A(160, 142, 240, 142)); b.append(R(240, 120, 140, 44, "grn", 6)); b.append(T(310, 147, "price database", "ink", "middle"))
    b.append(T(210, 186, "(it never touches anything)", "hand", "middle"))
    b.append(T(440, 104, "how it works", "h2"))
    lanes = [(480, "your code", "yel"), (690, "LLM API", "pnk"), (880, "tool / DB", "grn")]
    for x, name, c in lanes:
        b.append(R(x - 60, 116, 120, 30, c, 4)); b.append(T(x, 136, name, "numb", "middle"))
        b.append(f'<line x1="{x}" y1="146" x2="{x}" y2="405" class="lnd"/>')
    steps = [(480, 690, 176, "1. messages + tool schemas (name, description, JSON params)"),
             (690, 480, 216, "2. finish_reason = tool_calls: get_ticket_price(\"London\")"),
             (480, 880, 256, "3. your code runs the function"),
             (880, 480, 296, "4. \"$799\""),
             (480, 690, 336, "5. history + {role: tool, tool_call_id, content: \"$799\"}"),
             (690, 480, 376, "6. finish_reason = stop: \"A return ticket is $799.\"")]
    for x1, x2, y, lab in steps:
        b.append(A(x1, y, x2 + (-8 if x2 > x1 else 8), y, "lnr" if x1 == 690 else "ln", "arr" if x1 == 690 else "ar"))
        b.append(T((x1 + x2) / 2, y - 6, lab, "num", "middle"))
    b.append(P("M440,336 C 410,330 410,222 440,216", "lnr", "arr")); b.append(T(404, 280, "loop while", "hand", "end")); b.append(T(404, 296, "tool_calls", "hand", "end"))
    b.append(note(40, 220, 290, 190, "Rules that bite",
                  ["• it may ask for SEVERAL tools at", "  once: run every call", "• it may ask again after results:", "  use while, not if; cap the loop", "• dispatch by a name-to-function dict,", "  never eval/globals()", "• tools are just described in the prompt"], "handk"))
    b.append(note(40, 430, 890, 120, "Same trick, bigger names",
                  ["Structured outputs, \"agents\", MCP servers and sub-agents are all this loop: the model emits a structured request,",
                   "code executes it, the result goes back into the history. An agent = an LLM running tools in a loop toward a goal.",
                   "Gradio note: its chat history drops tool messages, so keep the full message list yourself if you need it later."]))
    svg("tool-calling.svg", 960, 570, "".join(b))

# ---------------- Week 3: two API levels + decoder stack ----------------
if want("hf"):
    b = [header("HUGGING FACE", "Pipelines on top, tokenizer + model underneath", 960)]
    b.append(note(30, 96, 300, 130, "pipeline(task, model)", ["one call: preprocess, run, decode", "sentiment, NER, QA, summarise,", "translate, zero-shot, generate,", "images (diffusers), speech"], "handk"))
    b.append(note(30, 240, 300, 150, "tokenizer + model", ["tok.apply_chat_template(msgs)", "model = AutoModelForCausalLM", "  .from_pretrained(name,", "  quantization_config=...)", "model.generate(ids, max_new_tokens)", "needed for fine-tuning"], "handk"))
    b.append(T(180, 418, "same model, two levels of control", "hand", "middle"))
    x0 = 400
    b.append(T(x0, 104, "inside SmolLM2-135M (printed shapes)", "h2"))
    rows = [("token IDs", "e.g. [1, 57, 1325, ...]", "cell"), ("embedding 49152 to 576", "one vector per token", "yel")]
    y = 122
    for name, sub, c in rows:
        b.append(R(x0, y, 300, 36, c, 4)); b.append(T(x0 + 10, y + 16, name, "numb")); b.append(T(x0 + 10, y + 30, sub, "mut")); b.append(A(x0 + 150, y + 36, x0 + 150, y + 48)); y += 48
    b.append(R(x0 - 10, y, 320, 196, "note", 8)); b.append(T(x0 + 300, y + 18, "× 30 layers", "hand", "end"))
    inner = [("RMSNorm", "grey"), ("self-attention: q 576x576, k/v 576 to 192, o", "blu"), ("+ residual, RMSNorm", "grey"),
             ("MLP: gate & up 576 to 1536, SiLU(gate)·up, down to 576", "grn"), ("+ residual", "grey")]
    yy = y + 26
    for name, c in inner:
        b.append(R(x0, yy, 300, 28, c, 3)); b.append(T(x0 + 10, yy + 18, name, "num")); yy += 34
    y += 206
    b.append(A(x0 + 150, y - 10, x0 + 150, y + 2))
    for name, sub, c in [("RMSNorm, lm_head 576 to 49152", "a score for every vocabulary entry", "pnk"), ("softmax: next-token probabilities", "pick one, append, repeat", "cell")]:
        b.append(R(x0, y, 300, 36, c, 4)); b.append(T(x0 + 10, y + 16, name, "numb")); b.append(T(x0 + 10, y + 30, sub, "mut")); b.append(A(x0 + 150, y + 36, x0 + 150, y + 46)); y += 46
    b.append(note(730, 96, 200, 200, "Reading it", ["attention: which earlier", "tokens matter here", "MLP: per-token feature", "transform (the bulk of", "the parameters)", "k/v narrower than q:", "grouped-query attention"], "handk"))
    b.append(note(730, 310, 200, 160, "Same shape, bigger", ["Llama 3.2 3B: 28 layers,", "3072 wide, MLP 8192,", "vocab 128,256.", "A 'next-token classifier'", "over the vocabulary."], "handk"))
    svg("hf-model-anatomy.svg", 960, 560, "".join(b))

# ---------------- Week 3: quantization levels ----------------
if want("hf"):
    from statistics import NormalDist
    b = [header("HUGGING FACE", "Quantization: fewer notches on the dimmer", 960)]
    X = lambda v: 480 + v * 400
    nd = NormalDist()
    pts = " ".join(f"{'M' if i == 0 else 'L'}{X(v / 100):.1f},{300 - 160 * nd.pdf(v / 100 * 2.5) / nd.pdf(0):.1f}" for i, v in enumerate(range(-100, 101)))
    b.append(P(pts, "ln")); b.append(T(480, 126, "weights in a layer are roughly bell-shaped (most near 0)", "mut", "middle"))
    b.append(P(f"M{X(-1)},300 L{X(1)},300", "ln"))
    even = [-1 + 2 * i / 15 for i in range(16)]
    nf = [nd.inv_cdf(0.02 + 0.96 * i / 15) for i in range(16)]; m = max(abs(v) for v in nf); nf = [v / m for v in nf]
    for v in even: b.append(f'<line x1="{X(v):.1f}" y1="330" x2="{X(v):.1f}" y2="356" class="ln" stroke-width="2"/>')
    b.append(T(X(-1), 324, "int4: 16 evenly spaced levels", "numb"))
    for v in nf: b.append(f'<line x1="{X(v):.1f}" y1="384" x2="{X(v):.1f}" y2="410" class="lnr" stroke-width="2"/>')
    b.append(T(X(-1), 378, "NF4: 16 levels at normal-distribution quantiles", "numb"))
    b.append(T(X(0), 436, "NF4 puts its notches where the weights are: denser near 0, sparser in the tails", "hand", "middle"))
    b.append(note(40, 456, 430, 100, "Memory for 3B parameters", ["fp32 12 GB · bf16 6 GB · 8-bit 3 GB · 4-bit 1.5 GB", "(+ small per-block scale factors; double quant", "compresses those too)"], "handk"))
    b.append(note(490, 456, 440, 100, "Why it works", ["each weight is stored as a 4-bit index plus a shared", "scale per block; compute de-quantizes to bf16.", "Accuracy drops a little, not 4× — known empirically."], "handk"))
    svg("quantization-levels.svg", 960, 570, "".join(b))

# ---------------- Week 4: choosing an LLM ----------------
if want("choose"):
    b = [header("CHOOSING LLMs", "From \"which is best?\" to \"which fits this job?\"", 960)]
    steps = [("1 understand", "business problem,", "success metric, data", "yel"), ("2 prepare", "basics → shortlist", "via leaderboards", "blu"),
             ("3 select", "prototype candidates,", "score on YOUR metric", "grn"), ("4 customise", "prompting, RAG,", "tools, fine-tuning", "pur"),
             ("5 productionise", "deploy, monitor,", "watch for drift", "pnk")]
    for k, (name, l1, l2, c) in enumerate(steps):
        x = 30 + k * 186
        b.append(R(x, 96, 170, 78, c, 6)); b.append(T(x + 85, 118, name, "h2", "middle"))
        b.append(T(x + 85, 140, l1.replace("→", "to"), "ink", "middle")); b.append(T(x + 85, 158, l2, "ink", "middle"))
        if k < 4: b.append(A(x + 171, 135, x + 185, 135))
    b.append(note(30, 192, 440, 214, "Basics (mostly on the model card)",
                  ["open-weight or closed · chat / reasoning / hybrid", "parameters · training tokens · context window", "release date and knowledge cutoff", "",
                   "Not on the card", "price per token (or GPU cost if self-hosted)", "speed (tokens/s) vs latency (time to first token)", "rate limits · licence terms · time to market"], "handk"))
    b.append(note(490, 192, 440, 214, "Benchmarks: useful, never final",
                  ["GPQA (448 PhD-level science Qs) · MMLU-Pro (10 options)", "AIME (contest maths) · LiveCodeBench (rotated)", "MuSR (murder mysteries) · HLE (~2,500 hardest Qs)", "",
                   "contamination: the test leaked into training", "saturation · self-reported runs · narrow scope", "small sets: ±4.5 points at 448 questions", "arenas: blind pairwise votes, Elo ratings"], "handk"))
    b.append(T(40, 434, "the bridge an engineer builds", "h2"))
    b.append(R(40, 448, 380, 100, "blu", 6)); b.append(T(230, 470, "model-centric", "h2", "middle"))
    b.append(T(230, 492, "loss · cross-entropy · perplexity", "ink", "middle")); b.append(T(230, 510, "precision · recall · F1 · AUC", "ink", "middle"))
    b.append(T(230, 532, "fast, optimisable, not what you sell", "hand", "middle"))
    b.append(R(540, 448, 380, 100, "grn", 6)); b.append(T(730, 470, "business-centric", "h2", "middle"))
    b.append(T(730, 492, "revenue · cost saved · time saved", "ink", "middle")); b.append(T(730, 510, "satisfaction (thumbs up/down)", "ink", "middle"))
    b.append(T(730, 532, "what matters, slow and noisy", "hand", "middle"))
    b.append(A(422, 498, 538, 498, "lnr", "arr")); b.append(A(538, 510, 422, 510, "lnr", "arr"))
    svg("choosing-llms.svg", 960, 570, "".join(b))

# ---------------- Week 5: RAG pipeline + evaluation ----------------
if want("rag"):
    b = [header("RAG", "Look it up, paste it in, then answer", 960)]
    def box(x, y, w, t1, t2, c):
        b.append(R(x, y, w, 46, c, 5)); b.append(T(x + w / 2, y + 19, t1, "numb", "middle")); b.append(T(x + w / 2, y + 36, t2, "mut", "middle"))
    b.append(T(40, 104, "ingest (once, offline)", "h2"))
    for k, (t1, t2, c) in enumerate([("documents", "md, pdf, wiki…", "cell"), ("chunk", "size + overlap", "cell"), ("encoder", "text to vector", "pur"), ("vector store", "chunk + vector + metadata", "grn")]):
        box(40 + k * 180, 118, 160, t1, t2, c)
        if k < 3: b.append(A(200 + k * 180, 141, 220 + k * 180, 141))
    b.append(T(40, 200, "query (every question)", "h2"))
    q = [("question", "+ chat history", "blu"), ("rewrite", "standalone query", "yel"), ("same encoder", "query vector", "pur"), ("top-k search", "nearest chunks", "grn"), ("rerank", "keep the best", "yel")]
    for k, (t1, t2, c) in enumerate(q):
        box(40 + k * 180, 214, 160, t1, t2, c)
        if k < 4: b.append(A(200 + k * 180, 237, 220 + k * 180, 237))
    b.append(P("M670,164 L670,214", "lnd", "ar")); b.append(T(678, 194, "searched", "mut"))
    b.append(P("M840,260 L840,300 L560,300", "ln", "ar"))
    b.append(R(250, 284, 300, 56, "pnk", 6)); b.append(T(400, 306, "prompt = instructions + chunks + question", "numb", "middle"))
    b.append(T(400, 326, "LLM answers from the pasted text (only text, never vectors)", "mut", "middle"))
    b.append(T(100, 312, "optional steps", "hand", "middle")); b.append(T(100, 328, "in yellow", "hand", "middle"))
    b.append(T(40, 372, "evaluate", "h2"))
    b.append(note(40, 386, 430, 170, "Retrieval (fast loop)", ["golden set: question + keywords + reference answer", "MRR = mean of 1 / rank of first relevant chunk", "nDCG = graded hits, discounted by log2(rank + 1)", "recall@k = share of questions with a hit in top k", "(recall matters most: the LLM can skip noise,", " it can't use what was never retrieved)"], "handk"))
    b.append(note(490, 386, 440, 170, "Answers (closer to the goal)", ["LLM-as-judge scores accuracy, completeness and", "relevance 1–5 against the reference answer", "use a strong judge; keep it fixed across runs", "", "change ONE thing per experiment:", "chunk size, encoder, k, rerank, prompt"], "handk"))
    svg("rag-pipeline.svg", 960, 575, "".join(b))

# ---------------- Weeks 6-8: the price-prediction leaderboard (course-reported MAE, $) ----------------
if want("price"):
    rows = [("random guess", 382.08, "grey", 6), ("constant: training mean", 106.18, "grey", 6), ("linear regression, 3 features", 101.56, "yel", 6),
            ("the instructor (100 items)", 87.62, "grey", 6), ("bag of words + linear", 76.81, "yel", 6), ("GPT-4.1-nano fine-tuned (20k)", 75.91, "pnk", 6),
            ("random forest", 72.28, "yel", 6), ("XGBoost", 68.23, "yel", 6), ("Llama 3.2 3B QLoRA, light run", 65.40, "grn", 7),
            ("vanilla neural net (669k params)", 63.97, "blu", 6), ("GPT-4.1-nano zero-shot", 62.51, "pnk", 6), ("Gemini 2.5 Flash-Lite", 58.68, "pnk", 6),
            ("Grok 4.1 Fast", 57.62, "pnk", 6), ("Gemini 3 Pro (50 items)", 50.54, "pnk", 6), ("Claude Opus 4.5", 47.10, "pnk", 6),
            ("deep residual net (289M params)", 46.49, "blu", 6), ("GPT-5.1", 44.74, "pnk", 6), ("Llama 3.2 3B QLoRA, full run", 39.85, "grn", 7),
            ("GPT-5.1 + RAG (5 similar items)", 30.19, "pur", 8), ("ensemble 0.8 RAG + 0.1 + 0.1", 29.90, "pur", 8)]
    b = [header("PRICE IS RIGHT", "Mean absolute error on held-out products (course-reported)", 960)]
    x0, y0, rh, scale = 300, 96, 22, 4.3
    for k, (name, v, c, wk) in enumerate(rows):
        y = y0 + k * rh
        b.append(T(x0 - 10, y + 15, name, "num", "end"))
        w = min(v, 120) * scale
        b.append(R(x0, y + 3, w, rh - 6, c, 2))
        if v > 120:
            b.append(f'<path d="M{x0 + w - 18},{y + 1} l6,{rh - 2} M{x0 + w - 12},{y + 1} l6,{rh - 2}" stroke="#f7f0e1" stroke-width="3"/>')
        b.append(T(x0 + w + 6, y + 15, f"${v:.2f}", "numb"))
        b.append(T(945, y + 15, f"wk {wk}", "mut", "end"))
    leg = [("grey", "baselines / human"), ("yel", "classical ML"), ("blu", "neural nets"), ("pnk", "frontier LLMs"), ("grn", "fine-tuned open LLM"), ("pur", "RAG + ensemble")]
    for k, (c, lab) in enumerate(leg):
        b.append(R(40 + k * 150, 582, 14, 14, c, 2)); b.append(T(60 + k * 150, 593, lab, "mut"))
    b.append(T(x0, 562, "lower is better · single runs on ~200 test items: differences of a few dollars are noise", "hand"))
    svg("price-leaderboard.svg", 960, 610, "".join(b))

# ---------------- Week 7: LoRA adapters + memory ----------------
if want("qlora"):
    b = [header("QLoRA", "Freeze the big matrix, train two thin ones beside it", 960)]
    b.append(T(40, 104, "1. one target layer, e.g. q_proj (3072 × 3072)", "h2"))
    b.append(R(60, 130, 150, 150, "grey", 4)); b.append(T(135, 200, "W (frozen)", "h2", "middle")); b.append(T(135, 220, "9.4M weights", "mut", "middle"))
    b.append(T(135, 300, "4-bit in QLoRA", "hand", "middle"))
    b.append(T(240, 210, "+", "h1", "middle"))
    b.append(R(270, 130, 22, 150, "yel", 3)); b.append(T(281, 300, "A", "h2", "middle")); b.append(T(281, 318, "3072×r", "mut", "middle"))
    b.append(T(306, 210, "·", "h1", "middle"))
    b.append(R(320, 194, 150, 22, "grn", 3)); b.append(T(395, 240, "B  r×3072", "h2", "middle")); b.append(T(395, 258, "starts at 0", "mut", "middle"))
    b.append(T(500, 210, "× α / r", "h2"))
    b.append(note(30, 336, 520, 120, "Why it works", ["the change a fine-tune needs is approximately low-rank:", "B·A has the full shape of W but only r·(in + out) numbers", "B = 0 at the start, so training begins from the base model", "r = 32: 196k numbers instead of 9.4M for this layer"], "handk"))
    b.append(T(600, 104, "2. hyperparameters", "h2"))
    b.append(note(590, 116, 340, 194, "QLoRA knobs", ["r (rank): 8–32 typical; 256 for the full run", "alpha: scale; alpha = 2r is common", "target modules: q, k, v, o first;", "  + gate, up, down for more capacity", "dropout on the adapter input: ~0.1", "base quantised to 4-bit NF4;", "  adapters stay in 16/32-bit"], "handk"))
    b.append(T(600, 340, "3. memory for Llama 3.2 3B", "h2"))
    bars = [("fp32 load", 12.9), ("8-bit", 3.6), ("4-bit NF4", 2.2), ("LoRA r=32 adapters", 0.073)]
    for k, (lab, gb) in enumerate(bars):
        y = 356 + k * 30
        b.append(R(720, y, max(gb * 13, 2), 20, "pnk" if k < 3 else "grn", 2)); b.append(T(710, y + 15, lab, "num", "end"))
        b.append(T(726 + gb * 13, y + 15, f"{gb:g} GB" if gb >= 1 else "73 MB", "numb"))
    b.append(note(30, 476, 900, 80, "Training then only updates the adapters",
                  ["Gradients and optimiser state exist for ~18M (light run) or ~389M (full run) parameters, not 3.2B —",
                   "that's what lets a 3B model fine-tune on a free 15 GB T4. Saved adapters are a separate small file loaded with PeftModel."]))
    svg("lora-adapters.svg", 960, 570, "".join(b))

# ---------------- Week 7: training curves (shape from the course's full run) ----------------
if want("qlora"):
    import math
    b = [header("QLoRA", "Watch validation loss, keep the best checkpoint", 960)]
    X = lambda s: 100 + s / 9375 * 760; Y = lambda l: 470 - (l - 1.0) * 600
    b.append(P(f"M{X(0)},{Y(1.0)} L{X(9375)},{Y(1.0)} M{X(0)},{Y(1.0)} L{X(0)},{Y(1.55)}", "ln"))
    for v in (1.1, 1.2, 1.3, 1.4, 1.5): b.append(T(X(0) - 8, Y(v) + 4, f"{v}", "num", "end"))
    for s, lab in ((0, "0"), (3125, "epoch 2"), (6250, "epoch 3"), (9375, "9,375 steps")): b.append(T(X(s), Y(1.0) + 18, lab, "mut", "middle"))
    for e in (3125, 6250): b.append(f'<line x1="{X(e)}" y1="{Y(1.0)}" x2="{X(e)}" y2="{Y(1.55)}" class="lnd"/>')
    tr = []
    for s in range(0, 9376, 75):
        e = s // 3125; base = 1.32 - 0.04 * (s / 9375) - 0.07 * e
        tr.append((s, base + 0.25 * math.exp(-s / 150) + 0.012 * math.sin(s / 37)))
    b.append(P(" ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(l):.1f}" for i, (s, l) in enumerate(tr)), "lnd"))
    ev = [(0, 1.27), (300, 1.31), (800, 1.22), (1600, 1.18), (2400, 1.16), (3125, 1.15), (4000, 1.14), (5000, 1.13), (6200, 1.12345), (6500, 1.20), (7500, 1.26), (8500, 1.28), (9375, 1.283)]
    b.append(P(" ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(l):.1f}" for i, (s, l) in enumerate(ev)), "lnr"))
    b.append(C(X(6200), Y(1.12345), 6, "grn")); b.append(T(X(6200) + 10, Y(1.12345) + 22, "best: 1.12345 at step 6,200", "hand"))
    b.append(T(X(300) + 8, Y(1.31) - 8, "spike at the LR peak", "hand"))
    b.append(T(X(7900), Y(1.29) - 10, "epoch 3: memorising", "hand", "middle"))
    b.append(T(X(4500), Y(1.27) - 6, "training loss (dashed): drops at each epoch boundary", "mut", "middle"))
    b.append(T(X(4500), Y(1.50), "validation loss (red) · schematic: shape and key values from the course's full run", "numb", "middle"))
    b.append(note(100, 500, 760, 60, "Rule", ["Save checkpoints every N steps; pick the one with the lowest validation loss (early stopping); then score the TEST set once."], "handk"))
    svg("qlora-training-curves.svg", 960, 575, "".join(b))

# ---------------- Week 8: the deal-finder agent architecture ----------------
if want("agents"):
    b = [header("AGENTIC AI", "Seven \"agents\": one planner loop, the rest are calls", 960)]
    def box(x, y, w, h, t1, t2, c, t3=None):
        b.append(R(x, y, w, h, c, 6)); b.append(T(x + w / 2, y + 20, t1, "h2", "middle")); b.append(T(x + w / 2, y + 38, t2, "mut", "middle"))
        if t3: b.append(T(x + w / 2, y + 54, t3, "mut", "middle"))
    box(330, 92, 300, 50, "UI (Gradio) + timer", "deals table, live logs, 3-D vector plot", "cell")
    box(330, 162, 300, 50, "agent framework", "memory (JSON of surfaced deals) + logging", "grey")
    box(330, 232, 300, 64, "planning agent", "LLM in a tool loop: scan, price,", "pnk", "pick the best discount, notify")
    for x1, y1, x2, y2 in ((480, 142, 480, 162), (480, 212, 480, 232)): b.append(A(x1, y1, x2, y2))
    subs = [(60, "scanner agent", "RSS feeds → LLM picks 5 deals", "structured outputs (Pydantic)", "yel"),
            (360, "ensemble agent", "fixed workflow, no autonomy", "0.8 RAG + 0.1 + 0.1", "blu"),
            (660, "messaging agent", "LLM writes the alert", "push notification", "grn")]
    for x, t1, t2, t3, c in subs:
        box(x, 330, 240, 70, t1.replace("→", "to"), t2.replace("→", "then"), c, t3)
        b.append(A(480, 296, x + 120, 330, "lnr", "arr"))
    leaves = [(140, "specialist agent", "fine-tuned Llama 3B", "served on Modal (GPU)"), (380, "frontier agent", "encode, search Chroma for 5", "similar items, then LLM (RAG)"),
              (620, "neural-net agent", "Week 6 deep", "residual network")]
    for x, t1, t2, t3 in leaves:
        box(x, 440, 200, 70, t1, t2.replace("→", ","), "blu", t3.replace("→", ""))
        b.append(A(480, 400, x + 100, 440))
    b.append(T(480, 540, "only the planner loops and chooses; the ensemble is code calling three models in a fixed order", "hand", "middle"))
    b.append(T(480, 558, "one full run: ~34 model calls across 6 different models", "mut", "middle"))
    svg("agent-architecture.svg", 960, 575, "".join(b))

# ---------------- AI A-Z: foundation-model lifecycle + where customisation happens ----------------
if want("genai"):
    b = [header("GENERATIVE AI", "A foundation model's life, and the cheap places to change it", 960)]
    steps = [("1 data", "collect, clean", "grey"), ("2 architecture", "choose / adapt", "grey"), ("3 pre-train", "next-token, $$$$", "pnk"),
             ("4 fine-tune", "domain, instructions, RLHF", "yel"), ("5 evaluate", "humans, metrics, judges", "blu"), ("6 deploy", "API, endpoint", "grn"),
             ("7 monitor", "drift, feedback", "grn"), ("8 iterate", "back to 1 or 4", "grey")]
    for k, (t1, t2, c) in enumerate(steps):
        x = 30 + (k % 4) * 232; y = 100 + (k // 4) * 96
        b.append(R(x, y, 210, 70, c, 6)); b.append(T(x + 105, y + 28, t1, "h2", "middle")); b.append(T(x + 105, y + 48, t2, "mut", "middle"))
        if k % 4 < 3: b.append(A(x + 211, y + 35, x + 231, y + 35))
    b.append(P("M830,170 L830,183 L135,183 L135,194", "ln", "ar"))
    b.append(T(40, 322, "cheapest to most expensive way to change behaviour", "h2"))
    lev = [("prompt", "4 elements, few-shot,", "chain-of-thought", "yel"), ("inference params", "temperature, top-p,", "top-k, max tokens", "yel"),
           ("RAG", "fetch facts at query", "time, cite sources", "blu"), ("fine-tune", "prompt/completion", "pairs, LoRA", "pur"), ("pre-train", "only for labs:", "data + GPUs + months", "pnk")]
    for k, (t1, t2, t3, c) in enumerate(lev):
        x = 30 + k * 186
        b.append(R(x, 336, 170, 84, c, 6)); b.append(T(x + 85, 360, t1, "h2", "middle")); b.append(T(x + 85, 380, t2, "mut", "middle")); b.append(T(x + 85, 396, t3, "mut", "middle"))
    b.append(A(40, 436, 920, 436, "lnr", "arr")); b.append(T(480, 456, "more cost, data and time; changes stick deeper", "hand", "middle"))
    b.append(note(30, 470, 900, 84, "Prompt anatomy (the course's four elements)",
                  ["instruction (what to do) · context (constraints, background) · input data (the material) · output indicator (format)",
                   "A template fixes all but the user's slots — and anything typed into a slot reaches the model, including injected instructions."]))
    svg("genai-lifecycle.svg", 960, 570, "".join(b))

# ---------------- AI A-Z: responsible AI ----------------
if want("rai"):
    b = [header("RESPONSIBLE AI", "Guard both directions, and know what you can explain", 960)]
    b.append(T(40, 104, "1. guardrails check the prompt AND the answer", "h2"))
    xs = [(40, "user", "cell"), (230, "guardrail", "yel"), (420, "model", "pnk"), (610, "guardrail", "yel"), (800, "user", "cell")]
    for x, t, c in xs:
        b.append(R(x, 124, 130, 50, c, 6)); b.append(T(x + 65, 154, t, "h2", "middle"))
    for x in (170, 360, 550, 740): b.append(A(x, 149, x + 60, 149))
    for x in (295, 675):
        b.append(P(f"M{x},174 L{x},206", "lnr", "arr")); b.append(T(x, 224, "blocked: fixed message", "hand", "middle"))
    b.append(T(480, 250, "content filters (hate, insults, sexual, violence, misconduct; adjustable strength) · denied topics · PII masking · grounding checks", "mut", "middle"))
    b.append(T(40, 292, "2. interpretability vs performance (typical, not a law)", "h2"))
    models = [("linear / logistic regression", 0.1), ("decision tree", 0.3), ("naive Bayes", 0.42), ("ensembles", 0.65), ("neural networks", 0.9)]
    b.append(A(60, 470, 900, 470)); b.append(A(60, 470, 60, 312)); b.append(T(900, 490, "performance on complex data", "mut", "end")); b.append(T(70, 322, "how transparent", "mut"))
    for name, v in models:
        x = 60 + v * 800; y = 330 + v * 120
        b.append(C(x, y, 7, "blu")); b.append(T(x + 12, y + 4, name, "num"))
    b.append(note(600, 290, 330, 106, "Two ways to answer \"why?\"", ["interpretability: read the model itself", "  (coefficients, a short tree)", "explainability: probe it from outside", "  (feature importance, partial dependence)"], "handk"))
    b.append(note(40, 500, 890, 56, "Responsible-AI checklist (course)", ["fairness · explainability · privacy and security · robustness · governance · transparency — plus legal risks: bias, hallucination, IP, toxicity, privacy"], "handk"))
    svg("responsible-ai.svg", 960, 570, "".join(b))
