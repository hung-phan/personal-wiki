# LLM Engineering

Study guides for building with large language models: own explanations, original diagrams (`assets/`, drawn with `scripts/llm_engineering_diagrams.py`), and snippets that run offline (a mock HTTP server stands in for paid APIs; small open models run on CPU). Neural-network foundations live in [deep-learning](../deep-learning/INDEX.md); applied retrieval and embeddings in [ml-systems](../ml-systems/INDEX.md); attention maths in [math-for-ml](../math-for-ml/INDEX.md). Up: [machine-learning](../INDEX.md).

## AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents (Udemy)

- [llm-foundations](llm-foundations.md) — Week 1: messages and roles, the OpenAI client as HTTP, OpenAI-compatible endpoints, tokens and tiktoken, stateless API and re-billed history, context windows, temperature, streaming, JSON mode and chained calls (brochure generator), base/chat/reasoning models, ways to run a model.
- [frontier-apis-and-tools](frontier-apis-and-tools.md) — Week 2: many providers from one client, routers vs abstraction layers, prompt caching by prefix, LLM-to-LLM conversations, Gradio callbacks and streaming, the tool-calling loop (parallel and sequential calls), multimodal airline assistant.
- [huggingface-open-models](huggingface-open-models.md) — Week 3: Hub and libraries, pipelines vs tokenizer + model, chat templates, next-token probabilities, the decoder stack printed layer by layer, 4-bit NF4 quantization, Colab GPUs, meeting-minutes project.
- [choosing-llms](choosing-llms.md) — Week 4: requirements-first selection, model-card basics and hidden costs, Chinchilla arithmetic, hard benchmarks and their error bars, leaderboards, arena Elo, dominated models, model vs business metrics, the C++/Rust porting showdown (algorithm beats language).
- [retrieval-augmented-generation](retrieval-augmented-generation.md) — Week 5: keyword toy vs embeddings, chunking and lost context, vector stores, query rewriting, golden-set evaluation (MRR, nDCG, recall@k, LLM judge), BM25 vs dense vs hybrid RRF vs cross-encoder rerank, ten advanced techniques.
- [price-prediction-and-frontier-fine-tuning](price-prediction-and-frontier-fine-tuning.md) — Week 6: data curation and leakage, weighted sampling, the baseline ladder, neural nets on hashed text, the evaluator, zero-shot frontier models, JSONL SFT of a frontier model and why it got worse; full price leaderboard.
- [qlora-fine-tuning](qlora-fine-tuning.md) — Week 7: LoRA from scratch, r/alpha/targets, the 18M/389M parameter arithmetic, 4-bit base, a real PEFT LoRA fine-tune on CPU, Keras enable_lora, warmup + cosine, validation-loss checkpointing, a 3B model beating the frontier.
- [agentic-ai-systems](agentic-ai-systems.md) — Week 8: agent definitions and the agentic trap, structured outputs and constrained decoding, ensembles (fitted vs hand weights), RAG pricer, Modal serving, planner loop with memory, injection and dispatch risks.

## AI A-Z (Udemy) — generative AI parts

- [generative-ai-and-prompting](generative-ai-and-prompting.md) — Parts 1–3 and 9: four elements of a prompt, templates and injection, few-shot and chain-of-thought, temperature/top-p/top-k worked, foundation-model lifecycle and selection, tokens and embeddings, transformers, fine-tuning types, ROUGE/BLEU by hand vs judges, RAG and agents on AWS Bedrock, the Llama 2 medical QLoRA fine-tune. (Reinforcement-learning parts: [reinforcement-learning](../reinforcement-learning/INDEX.md).)
- [responsible-ai](responsible-ai.md) — Part 10: six features, two-way guardrails and Bedrock Guardrails, five legal risks, bias by selection rates and the four-fifths rule, Clarify / Model Monitor / A2I / Model Cards, interpretability vs explainability with partial dependence.
