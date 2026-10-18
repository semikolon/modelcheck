//! modelcheck — fast one-shot cross-provider LLM model info lookup.
//!
//! Fans out concurrently to no-key public catalogs and merges the results:
//!   - models.dev      → pricing (in/out/cache) + context + capabilities (per-million USD)
//!   - OpenRouter      → cross-provider availability + context + pricing (per-token)
//!   - Artificial Analysis (opt-in via AA_API_KEY) → tok/s, TTFT, intelligence index
//!
//! Each source is independently timed-out and failure-tolerant: a dead source shows
//! "n/a" and never crashes the whole command.
//!
//! Usage:  modelcheck <name-substring> [--json]
//!   e.g.  modelcheck gemini-3.5-flash
//!         modelcheck opus --json

use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;

const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/models";
const AA_URL: &str = "https://artificialanalysis.ai/api/v2/language/models/free";
const TIMEOUT_SECS: u64 = 8;

// ---------- ANSI (only when stdout is a TTY) ----------
struct Style {
    on: bool,
}
impl Style {
    fn new() -> Self {
        // std::io::IsTerminal is stable since 1.70
        use std::io::IsTerminal;
        Style {
            on: std::io::stdout().is_terminal(),
        }
    }
    fn bold(&self, s: &str) -> String {
        if self.on {
            format!("\x1b[1m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    fn cyan(&self, s: &str) -> String {
        if self.on {
            format!("\x1b[36m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    fn green(&self, s: &str) -> String {
        if self.on {
            format!("\x1b[32m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    fn dim(&self, s: &str) -> String {
        if self.on {
            format!("\x1b[2m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    fn yellow(&self, s: &str) -> String {
        if self.on {
            format!("\x1b[33m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
}

// First-party / primary providers — preferred as the canonical source of a
// model's metadata over resellers/gateways (aihubmix, cortecs, digitalocean…).
const FIRST_PARTY: &[&str] = &[
    "anthropic",
    "openai",
    "google",
    "google-ai-studio",
    "google-vertex",
    "xai",
    "mistral",
    "meta",
    "deepseek",
    "amazon-bedrock",
    "azure",
];

fn first_party_rank(provider: &str) -> usize {
    // lower = more canonical
    FIRST_PARTY
        .iter()
        .position(|p| *p == provider)
        .unwrap_or(usize::MAX)
}

// Collapse a model id to a canonical identity for deduping:
// strip provider prefix, drop region/mode/format suffixes, normalize "." → "-".
fn canonical_id(raw: &str) -> String {
    let bare = raw.rsplit('/').next().unwrap_or(raw);
    // strip common bedrock/vertex region and reseller decorations
    let bare = bare
        .trim_start_matches("us.")
        .trim_start_matches("eu.")
        .trim_start_matches("jp.")
        .trim_start_matches("au.")
        .trim_start_matches("global.")
        .trim_start_matches("anthropic.")
        .trim_start_matches("amazon.");
    // drop trailing decorations after @ or :
    let bare = bare.split(['@', ':']).next().unwrap_or(bare);
    let mut c = bare.replace('.', "-").to_lowercase();
    // strip a provider name embedded as a prefix in the id itself
    // (e.g. digitalocean "anthropic-claude-opus-4-8", SAP "anthropic--claude-3-opus")
    for p in [
        "anthropic",
        "google",
        "openai",
        "amazon",
        "meta",
        "mistralai",
        "databricks",
    ] {
        if let Some(rest) = c.strip_prefix(p) {
            // only when a separator follows, so "openai-o3" strips but "meta-llama" is intact-ish
            let rest = rest.trim_start_matches('-');
            if !rest.is_empty() && rest != c {
                c = rest.to_string();
                break;
            }
        }
    }
    c.trim_matches('-').to_string()
}

// ---------- Merged per-model record ----------
#[derive(Default, Clone)]
struct ModelRow {
    // identity
    id: String,           // canonical-ish id for the row header
    display_name: String, // human name (best available)
    // availability
    on_models_dev: bool,
    md_provider: String,      // best (most-canonical) models.dev provider seen
    md_provider_rank: usize,  // rank of md_provider (lower = more canonical)
    md_provider_count: usize, // how many models.dev providers carry this model
    on_openrouter: bool,
    // context / limits
    context: Option<u64>,
    max_output: Option<u64>,
    // pricing (per-million USD) — preferred from models.dev, fallback OpenRouter
    price_in: Option<f64>,
    price_out: Option<f64>,
    price_cache_read: Option<f64>,
    price_cache_write: Option<f64>,
    price_source: String, // "models.dev" | "openrouter" | ""
    // capabilities
    reasoning: Option<bool>,
    tool_call: Option<bool>,
    modalities_in: Vec<String>,
    release_date: Option<String>,
    // stats (Artificial Analysis, opt-in)
    tokens_per_sec: Option<f64>,
    ttft: Option<f64>,
    intelligence: Option<f64>,
}

impl ModelRow {
    // Deterministic ordering key: the id that IS the query first, then
    // first-party rows before resellers, then alphabetical.
    fn sort_key(&self, needle: &str) -> (bool, usize, String) {
        let id = self.id.to_lowercase();
        let bare = norm(&id);
        let bare = bare.strip_suffix(" preview").unwrap_or(&bare);
        (bare != needle, self.md_provider_rank, id)
    }
}

fn pt_to_per_m(s: &str) -> Option<f64> {
    // OpenRouter pricing values are per-token strings; ×1e6 → per-million.
    s.parse::<f64>().ok().map(|v| v * 1_000_000.0)
}

fn as_u64(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_f64().map(|f| f as u64))
}

fn as_f64_or_str(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

// ---------- fetchers (each returns Result so a failure is isolated) ----------
async fn fetch_json(
    client: &reqwest::Client,
    url: &str,
    aa_key: Option<&str>,
) -> Result<Value, String> {
    let mut req = client.get(url).timeout(Duration::from_secs(TIMEOUT_SECS));
    if let Some(k) = aa_key {
        req = req.header("x-api-key", k);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status().as_u16()));
    }
    resp.json::<Value>().await.map_err(|e| e.to_string())
}

// Substring match, normalizing separators so "gemini-3.5-flash" matches
// "gemini-3.5-flash", "google/gemini-3.5-flash", "gemini_3_5_flash", etc.
fn norm(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| {
            if c == '-' || c == '_' || c == '/' || c == '.' {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
fn matches(haystack: &str, needle_norm: &str) -> bool {
    norm(haystack).contains(needle_norm)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json_mode = args.iter().any(|a| a == "--json");
    let all_mode = args.iter().any(|a| a == "--all" || a == "-a");
    let query: Vec<&str> = args
        .iter()
        .filter(|a| !a.starts_with("--") && *a != "-a")
        .map(|s| s.as_str())
        .collect();

    if query.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "modelcheck — one-shot cross-provider LLM model lookup\n\
             \nUSAGE:\n  modelcheck <name-substring> [--all] [--json]\n\
             \nEXAMPLES:\n  modelcheck gemini-3.5-flash\n  modelcheck opus\n  modelcheck claude-opus-4-8 --json\n  modelcheck opus --all      # every per-provider variant (bedrock regions, resellers…)\n\
             \nBy default one canonical row per model (first-party provider preferred).\n\
             \nSOURCES (no key needed):\n  models.dev   pricing + context + capabilities\n  openrouter   cross-provider availability + pricing\n  artificial-analysis  tok/s + TTFT + intelligence (opt-in: set AA_API_KEY)\n"
        );
        std::process::exit(if query.is_empty() { 2 } else { 0 });
    }
    let needle = norm(&query.join(" "));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let exit = rt.block_on(run(needle, json_mode, all_mode));
    std::process::exit(exit);
}

async fn run(needle: String, json_mode: bool, all_mode: bool) -> i32 {
    let client = reqwest::Client::builder()
        .user_agent("modelcheck/0.1 (+https://github.com/semikolon/modelcheck)")
        .build()
        .expect("http client");
    let aa_key = std::env::var("AA_API_KEY").ok().filter(|k| !k.is_empty());

    // Fan out concurrently.
    let md_fut = fetch_json(&client, MODELS_DEV_URL, None);
    let or_fut = fetch_json(&client, OPENROUTER_URL, None);
    let aa_fut = async {
        match &aa_key {
            Some(k) => Some(fetch_json(&client, AA_URL, Some(k)).await),
            None => None,
        }
    };
    let (md_res, or_res, aa_res) = tokio::join!(md_fut, or_fut, aa_fut);

    let style = Style::new();
    let mut source_errors: Vec<String> = Vec::new();

    // Default: rows keyed by canonical model identity (dedup across providers,
    // first-party preferred). --all: keyed by raw id (every per-provider variant).
    let mut rows: BTreeMap<String, ModelRow> = BTreeMap::new();

    // ---- models.dev ----
    match &md_res {
        Ok(v) => merge_models_dev(v, &needle, all_mode, &mut rows),
        Err(e) => source_errors.push(format!("models.dev: {e}")),
    }
    // ---- OpenRouter ----
    match &or_res {
        Ok(v) => merge_openrouter(v, &needle, all_mode, &mut rows),
        Err(e) => source_errors.push(format!("openrouter: {e}")),
    }
    // ---- Artificial Analysis (opt-in) ----
    match &aa_res {
        Some(Ok(v)) => merge_aa(v, &needle, &mut rows),
        Some(Err(e)) => source_errors.push(format!("artificial-analysis: {e}")),
        None => {}
    }

    if !all_mode {
        fold_preview_twins(&mut rows);
    }
    let mut list: Vec<ModelRow> = rows.into_values().collect();
    list.sort_by_key(|a| a.sort_key(&needle));

    if json_mode {
        print_json(&list, &source_errors, aa_key.is_some());
        return if list.is_empty() { 1 } else { 0 };
    }

    if list.is_empty() {
        eprintln!("{} no models matched \"{}\"", style.yellow("✗"), needle);
        if !source_errors.is_empty() {
            eprintln!(
                "{} sources unavailable: {}",
                style.dim("·"),
                source_errors.join(", ")
            );
        } else {
            eprintln!(
                "{} try a shorter substring, e.g. `modelcheck opus`",
                style.dim("·")
            );
        }
        return 1;
    }

    for row in &list {
        print_row(row, &style);
    }

    if !source_errors.is_empty() {
        eprintln!(
            "\n{} some sources failed: {}",
            style.yellow("⚠"),
            source_errors.join(", ")
        );
    }
    if aa_key.is_none() {
        eprintln!(
            "{}",
            style.dim("· stats (tok/s, TTFT) not shown — set AA_API_KEY for Artificial Analysis")
        );
    }
    0
}

// A models.dev id often lacks the "-preview" suffix that OpenRouter carries
// ("gemini-3-1-pro" vs "google/gemini-3.1-pro-preview"). When the exact key is
// absent but the suffix-stripped one already exists, attach to that row.
fn attach_key(key: String, all_mode: bool, rows: &BTreeMap<String, ModelRow>) -> String {
    if all_mode || rows.contains_key(&key) {
        return key;
    }
    match key.strip_suffix("-preview") {
        Some(base) if rows.contains_key(base) => base.to_string(),
        _ => key,
    }
}

// A reseller lists "gemini-3-1-pro" while Google lists "gemini-3-1-pro-preview":
// one model under two keys. Fold the reseller-only key into the first-party
// preview row so one row answers the query.
fn fold_preview_twins(rows: &mut BTreeMap<String, ModelRow>) {
    let keys: Vec<String> = rows.keys().cloned().collect();
    for k in keys {
        let twin = format!("{k}-preview");
        let reseller_only = rows
            .get(&k)
            .map(|r| r.md_provider_rank == usize::MAX)
            .unwrap_or(false);
        if reseller_only && rows.contains_key(&twin) {
            if let Some(r) = rows.remove(&k) {
                let t = rows.get_mut(&twin).expect("twin exists");
                t.md_provider_count += r.md_provider_count;
                t.on_openrouter |= r.on_openrouter;
            }
        }
    }
}

// ---------- merge: models.dev ----------
fn merge_models_dev(
    root: &Value,
    needle: &str,
    all_mode: bool,
    rows: &mut BTreeMap<String, ModelRow>,
) {
    let Some(providers) = root.as_object() else {
        return;
    };
    for (prov_key, prov) in providers {
        let Some(models) = prov.get("models").and_then(|m| m.as_object()) else {
            continue;
        };
        for (model_id, m) in models {
            let name = m.get("name").and_then(|n| n.as_str()).unwrap_or("");
            if !(matches(model_id, needle) || matches(name, needle)) {
                continue;
            }
            // Default view: one row per canonical model. --all: one row per raw id.
            let key = if all_mode {
                format!("{prov_key}/{model_id}").to_lowercase()
            } else {
                canonical_id(model_id)
            };
            let rank = first_party_rank(prov_key);
            let row = rows.entry(key).or_insert_with(|| ModelRow {
                md_provider_rank: usize::MAX,
                ..Default::default()
            });
            row.on_models_dev = true;
            row.md_provider_count += 1;

            // In default mode, keep metadata from the MOST-canonical provider.
            // First provider for this key, or a strictly more-canonical one, wins.
            let more_canonical = rank < row.md_provider_rank;
            let first_fill = row.md_provider.is_empty();
            if first_fill || more_canonical || all_mode {
                row.id = if all_mode {
                    model_id.clone()
                } else {
                    canonical_id(model_id)
                };
                row.display_name = name.to_string();
                row.md_provider = prov_key.clone();
                row.md_provider_rank = rank;
                if let Some(limit) = m.get("limit") {
                    row.context = limit.get("context").and_then(as_u64);
                    row.max_output = limit.get("output").and_then(as_u64);
                }
                if let Some(cost) = m.get("cost") {
                    row.price_in = cost.get("input").and_then(|v| v.as_f64());
                    row.price_out = cost.get("output").and_then(|v| v.as_f64());
                    row.price_cache_read = cost.get("cache_read").and_then(|v| v.as_f64());
                    row.price_cache_write = cost.get("cache_write").and_then(|v| v.as_f64());
                    if row.price_in.is_some() || row.price_out.is_some() {
                        row.price_source = "models.dev".to_string();
                    } else {
                        row.price_source.clear();
                    }
                }
                row.reasoning = m.get("reasoning").and_then(|v| v.as_bool());
                row.tool_call = m.get("tool_call").and_then(|v| v.as_bool());
                row.modalities_in = m
                    .get("modalities")
                    .and_then(|x| x.get("input"))
                    .and_then(|x| x.as_array())
                    .map(|mods| {
                        mods.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                row.release_date = m
                    .get("release_date")
                    .and_then(|v| v.as_str())
                    .map(String::from);
            }
        }
    }
}

// ---------- merge: OpenRouter ----------
fn merge_openrouter(
    root: &Value,
    needle: &str,
    all_mode: bool,
    rows: &mut BTreeMap<String, ModelRow>,
) {
    let Some(data) = root.get("data").and_then(|d| d.as_array()) else {
        return;
    };
    for m in data {
        let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let name = m.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let canonical = m
            .get("canonical_slug")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !(matches(id, needle) || matches(name, needle) || matches(canonical, needle)) {
            continue;
        }
        // Key by canonical model identity so OR availability attaches to the
        // models.dev row for the same model. In --all mode keep the raw OR id.
        let key = if all_mode {
            id.to_lowercase()
        } else {
            canonical_id(id)
        };
        let key = attach_key(key, all_mode, rows);
        let row = rows.entry(key).or_insert_with(|| ModelRow {
            md_provider_rank: usize::MAX,
            ..Default::default()
        });
        if row.id.is_empty() {
            row.id = if all_mode {
                id.to_string()
            } else {
                canonical_id(id)
            };
        }
        if row.display_name.is_empty() {
            row.display_name = name.to_string();
        }
        row.on_openrouter = true;
        if row.context.is_none() {
            row.context = m.get("context_length").and_then(as_u64);
        }
        if row.max_output.is_none() {
            row.max_output = m
                .get("top_provider")
                .and_then(|t| t.get("max_completion_tokens"))
                .and_then(as_u64);
        }
        // Only fill pricing from OR if models.dev didn't provide it.
        if row.price_source.is_empty() {
            if let Some(p) = m.get("pricing") {
                row.price_in = p
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .and_then(pt_to_per_m);
                row.price_out = p
                    .get("completion")
                    .and_then(|v| v.as_str())
                    .and_then(pt_to_per_m);
                row.price_cache_read = p
                    .get("input_cache_read")
                    .and_then(|v| v.as_str())
                    .and_then(pt_to_per_m);
                row.price_cache_write = p
                    .get("input_cache_write")
                    .and_then(|v| v.as_str())
                    .and_then(pt_to_per_m);
                if row.price_in.is_some() || row.price_out.is_some() {
                    row.price_source = "openrouter".to_string();
                }
            }
        }
        if row.modalities_in.is_empty() {
            if let Some(mods) = m
                .get("architecture")
                .and_then(|a| a.get("input_modalities"))
                .and_then(|x| x.as_array())
            {
                row.modalities_in = mods
                    .iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect();
            }
        }
    }
}

// ---------- merge: Artificial Analysis (opt-in) ----------
fn merge_aa(root: &Value, needle: &str, rows: &mut BTreeMap<String, ModelRow>) {
    // AA free endpoint returns { "data": [ {...} ] } (or a bare array on some tiers).
    let arr = root
        .get("data")
        .and_then(|d| d.as_array())
        .cloned()
        .or_else(|| root.as_array().cloned())
        .unwrap_or_default();
    for m in &arr {
        let name = m.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let slug = m.get("slug").and_then(|v| v.as_str()).unwrap_or("");
        let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if !(matches(name, needle) || matches(slug, needle) || matches(id, needle)) {
            continue;
        }
        // Attach stats to any already-known row whose id/name matches this AA entry.
        let tps = m
            .get("median_output_tokens_per_second")
            .and_then(as_f64_or_str);
        let ttft = m
            .get("median_time_to_first_token_seconds")
            .and_then(as_f64_or_str);
        let intel = m
            .get("evaluations")
            .and_then(|e| e.get("artificial_analysis_intelligence_index"))
            .and_then(as_f64_or_str)
            .or_else(|| {
                m.get("artificial_analysis_intelligence_index")
                    .and_then(as_f64_or_str)
            });

        let mut attached = false;
        for row in rows.values_mut() {
            if matches(&row.id, &norm(name))
                || matches(name, &norm(&row.id))
                || (!slug.is_empty() && matches(&row.id, &norm(slug)))
            {
                row.tokens_per_sec = tps;
                row.ttft = ttft;
                row.intelligence = intel;
                attached = true;
            }
        }
        if !attached {
            // Own row so the stats aren't lost.
            let key = format!("aa:{}", slug.to_lowercase());
            let row = rows.entry(key).or_default();
            if row.id.is_empty() {
                row.id = if !slug.is_empty() {
                    slug.to_string()
                } else {
                    name.to_string()
                };
            }
            if row.display_name.is_empty() {
                row.display_name = name.to_string();
            }
            row.tokens_per_sec = tps;
            row.ttft = ttft;
            row.intelligence = intel;
        }
    }
}

// ---------- output ----------
fn fmt_price(p: Option<f64>) -> String {
    match p {
        Some(v) => {
            // trim trailing zeros for readability
            let s = format!("{:.4}", v);
            let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
            format!("${s}")
        }
        None => "n/a".to_string(),
    }
}
fn fmt_ctx(c: Option<u64>) -> String {
    match c {
        Some(v) if v >= 1_000_000 => format!("{}M", v / 1_000_000),
        Some(v) if v >= 1_000 => format!("{}k", v / 1_000),
        Some(v) => v.to_string(),
        None => "n/a".to_string(),
    }
}
fn avail(b: bool, style: &Style) -> String {
    if b {
        style.green("✓")
    } else {
        style.dim("✗")
    }
}

fn print_row(r: &ModelRow, style: &Style) {
    let title = if r.display_name.is_empty() {
        r.id.clone()
    } else {
        format!("{}  {}", r.id, style.dim(&format!("({})", r.display_name)))
    };
    println!("{}", style.bold(&style.cyan(&title)));

    // availability line
    let mut avails = vec![
        format!("models.dev {}", avail(r.on_models_dev, style)),
        format!("openrouter {}", avail(r.on_openrouter, style)),
    ];
    if !r.md_provider.is_empty() {
        let extra = if r.md_provider_count > 1 {
            format!(" +{} more", r.md_provider_count - 1)
        } else {
            String::new()
        };
        avails.push(style.dim(&format!("via {}{}", r.md_provider, extra)));
    }
    println!("  {}", avails.join("   "));

    // context line
    println!(
        "  ctx {}   max-out {}",
        fmt_ctx(r.context),
        fmt_ctx(r.max_output)
    );

    // pricing line (per 1M tokens USD)
    let src = if r.price_source.is_empty() {
        String::new()
    } else {
        style.dim(&format!("  [{}]", r.price_source))
    };
    println!(
        "  in {} / out {}   cache-read {}  cache-write {}{}",
        fmt_price(r.price_in),
        fmt_price(r.price_out),
        fmt_price(r.price_cache_read),
        fmt_price(r.price_cache_write),
        src
    );

    // capabilities line
    let mut caps: Vec<String> = Vec::new();
    if r.reasoning == Some(true) {
        caps.push("reasoning".into());
    }
    if r.tool_call == Some(true) {
        caps.push("tools".into());
    }
    if !r.modalities_in.is_empty() {
        caps.push(format!("in: {}", r.modalities_in.join("+")));
    }
    if let Some(rd) = &r.release_date {
        caps.push(format!("released {rd}"));
    }
    if !caps.is_empty() {
        println!("  {}", style.dim(&caps.join("   ")));
    }

    // stats line (AA)
    if r.tokens_per_sec.is_some() || r.ttft.is_some() || r.intelligence.is_some() {
        let mut stats = Vec::new();
        if let Some(t) = r.tokens_per_sec {
            stats.push(format!("{:.0} tok/s", t));
        }
        if let Some(t) = r.ttft {
            stats.push(format!("TTFT {:.2}s", t));
        }
        if let Some(i) = r.intelligence {
            stats.push(format!("intelligence {:.1}", i));
        }
        println!("  {}", style.yellow(&stats.join("   ")));
    }
    println!();
}

fn print_json(list: &[ModelRow], errors: &[String], aa_enabled: bool) {
    let arr: Vec<Value> = list
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id,
                "display_name": r.display_name,
                "availability": {
                    "models_dev": r.on_models_dev,
                    "openrouter": r.on_openrouter,
                    "models_dev_provider": r.md_provider,
                    "models_dev_provider_count": r.md_provider_count,
                },
                "context": r.context,
                "max_output": r.max_output,
                "pricing_per_million_usd": {
                    "input": r.price_in,
                    "output": r.price_out,
                    "cache_read": r.price_cache_read,
                    "cache_write": r.price_cache_write,
                    "source": r.price_source,
                },
                "capabilities": {
                    "reasoning": r.reasoning,
                    "tool_call": r.tool_call,
                    "input_modalities": r.modalities_in,
                    "release_date": r.release_date,
                },
                "stats": {
                    "tokens_per_sec": r.tokens_per_sec,
                    "ttft_seconds": r.ttft,
                    "intelligence_index": r.intelligence,
                }
            })
        })
        .collect();
    let out = serde_json::json!({
        "query_matched": list.len(),
        "aa_enabled": aa_enabled,
        "source_errors": errors,
        "models": arr,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reseller_prefix_folds_into_the_canonical_id() {
        assert_eq!(canonical_id("databricks-gemini-3-5-flash"), "gemini-3-5-flash");
        assert_eq!(canonical_id("google/gemini-3.5-flash"), "gemini-3-5-flash");
    }

    #[test]
    fn the_queried_model_sorts_before_lookalikes_and_resellers() {
        let needle = norm("gemini-3.5-flash");
        let row = |id: &str, rank: usize| ModelRow {
            id: id.to_string(),
            md_provider_rank: rank,
            ..Default::default()
        };
        let mut v = vec![
            row("aihubmix-gemini-3-5-flash", usize::MAX),
            row("gemini-3-5-flash-lite", 2),
            row("gemini-3-5-flash", 2),
        ];
        v.sort_by_key(|r| r.sort_key(&needle));
        let ids: Vec<&str> = v.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["gemini-3-5-flash", "gemini-3-5-flash-lite", "aihubmix-gemini-3-5-flash"]
        );
    }

    #[test]
    fn a_reseller_only_key_folds_into_the_first_party_preview_row() {
        let mut rows: BTreeMap<String, ModelRow> = BTreeMap::new();
        rows.insert("gemini-3-1-pro".into(), ModelRow { md_provider_rank: usize::MAX, md_provider_count: 3, ..Default::default() });
        rows.insert("gemini-3-1-pro-preview".into(), ModelRow { md_provider_rank: 2, md_provider_count: 1, ..Default::default() });
        fold_preview_twins(&mut rows);
        assert!(!rows.contains_key("gemini-3-1-pro"));
        assert_eq!(rows["gemini-3-1-pro-preview"].md_provider_count, 4);
        // the query "gemini-3.1-pro" still counts the preview row as the exact hit
        let needle = norm("gemini-3.1-pro");
        let row = ModelRow { id: "gemini-3-1-pro-preview".into(), md_provider_rank: 2, ..Default::default() };
        assert!(!row.sort_key(&needle).0);
    }

    #[test]
    fn an_openrouter_preview_id_attaches_to_the_suffixless_row() {
        let mut rows: BTreeMap<String, ModelRow> = BTreeMap::new();
        rows.insert("gemini-3-1-pro".into(), ModelRow::default());
        assert_eq!(attach_key("gemini-3-1-pro-preview".into(), false, &rows), "gemini-3-1-pro");
        // a preview that models.dev also lists stays its own row
        rows.insert("gemini-3-flash-preview".into(), ModelRow::default());
        assert_eq!(attach_key("gemini-3-flash-preview".into(), false, &rows), "gemini-3-flash-preview");
        // --all keeps every raw id apart
        assert_eq!(attach_key("gemini-3-1-pro-preview".into(), true, &rows), "gemini-3-1-pro-preview");
    }
}
