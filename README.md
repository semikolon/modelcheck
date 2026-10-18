# modelcheck

Fast, one-shot cross-provider LLM model lookup. Kills the *"wrongly assumed a
model unavailable / wrong price / wrong context window"* failure class by
merging several authoritative no-key catalogs into one answer.

```
modelcheck gemini-3.5-flash
modelcheck claude-opus-4-8
modelcheck opus --all      # every per-provider variant (Bedrock regions, resellers…)
modelcheck opus --json     # machine output
```

## What it does

Fans out **concurrently** (tokio + reqwest) to three sources and merges them
into one row per model:

| Source | Contributes | Key needed |
|---|---|---|
| [models.dev](https://models.dev) `api.json` | pricing (in/out/cache, per-million USD) + context + capabilities | no |
| [OpenRouter](https://openrouter.ai) `/api/v1/models` | cross-provider availability + context + pricing | no |
| [Artificial Analysis](https://artificialanalysis.ai) free API | tok/s, TTFT, intelligence index | opt-in: `AA_API_KEY` |

- **Default view**: one canonical row per model, first-party provider preferred
  (Anthropic/OpenAI/Google over resellers). Shows `via <provider> +N more`.
  The model you named sorts first, then first-party rows, then the rest;
  reseller-prefixed ids (`databricks-gemini-3-5-flash`) fold into the canonical
  row, and an OpenRouter `-preview` id attaches to the models.dev row when that
  one carries no suffix (`gemini-3-1-pro` ⇐ `google/gemini-3.1-pro-preview`).
- **`--all`**: every per-provider variant (Bedrock regional profiles, gateways).
- **Failure-tolerant**: each source is independently timed out (8s); a dead
  source shows `n/a`, never crashes the command.
- **`--json`**: structured output for scripting.

## Output fields

```
claude-opus-4-8  (Claude Opus 4.8)
  models.dev ✓   openrouter ✓   via anthropic +29 more
  ctx 1M   max-out 128k
  in $5 / out $25   cache-read $0.5  cache-write $6.25  [models.dev]
  reasoning   tools   in: text+image+pdf   released 2026-05-28
```

## Install

```
cargo build --release
cp target/release/modelcheck ~/.local/bin/
```

## Caveats

- Aggregators can trail a provider's own announcement by hours-to-days at a
  brand-new model's release. For a budget-critical decision on a just-released
  model, confirm against the provider's own pricing page.
- **These aggregators tell you a model *exists*; only a provider-native list
  (`GET /v1/models` on your key) tells you it's callable on *your* key/tier/
  region.** modelcheck deliberately does not touch your keys.
- The 150+ catalogs name the same model differently (`claude-opus-4-6` vs
  `claude-4-6-opus` vs dated `-20251101` variants). An exact-id query gives a
  clean answer; a broad substring shows the naming variants honestly rather
  than risk merging genuinely-distinct models.

Prior art (none does the merged one-shot lookup): `llm-pricing`
(OpenRouter-only), `llmkit` (baked-in registry client), `llm-cost-cap` (a
library). Research: `~/dotfiles/docs/llm_model_info_tools_research_2026_07_03.md`.
