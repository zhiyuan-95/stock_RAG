====
summarize commands in one sentence, write down the major change, and add a date heading the first time the log is updated on that day; if the total code change adds and removes more than 1000 lines, break it down into steps
===

# Command Summary Log

## 2026-03-25

1. Updated `ingest_knowledge.py` so glossary documents load from `data_store/glossary/raw` when present and added a catalog helper that lists the 48 glossary indicators by group and subgroup.

2. Updated `ingest_stock.py` to compute the 48 glossary indicators from SEC Company Facts and the current stock price, store them in the stock SQL table, and rebuild the stock index from that SEC-derived dataset.

3. Removed the dead relationship-graph code from `ingest_stock.py` and `query.py` and changed stock refresh to always rewrite the current-price-based indicator rows before rebuilding the ticker index.

4. Updated `ingest_knowledge.py` to load glossary markdown from `data_store/glossary/company/raw` and `data_store/glossary/eco/raw` and generated 14 macro glossary files under `data_store/glossary/eco/raw` with CFI-based section 3 notes.

5. Updated `ingest_knowledge.py` so macro glossary files in `data_store/glossary/eco/raw` now store raw scraped CFI article text without the section 1-5 wrapper and rebuilt the knowledge index with the new format.

6. Updated `ingest_macro.py` so macro ingestion now only refreshes the eco-glossary-backed indicator set, prunes unsupported macro rows from `macro_data.db`, and uses a single glossary-backed `PMI` series.

7. Added a unified graph layer that links glossary concepts, stock filings, and the latest stock and macro observations into a graph index used alongside the existing stock, macro, and knowledge retrieval paths.

8. Replaced the old custom graph path with a per-ticker `PropertyGraphIndex + SimplePropertyGraphStore` design that builds graph-ready docs from glossary text, recent SEC filing text, and latest stock and macro observation snapshots.

9. Completed end-to-end runtime validation by fixing UTF-8 graph loading and adding a direct SQL financial-indicator supplement in `query.py`, which now lets `main.py` return an AAPL analysis with recent financial metrics, macro context, and graph-backed retrieval.

10. Refactored the stock filing document layer so all 10 years of `10-K`s and 12 quarters of `10-Q`s remain in the index with active/archive tiers, section summaries, and structured financial-statement docs replacing raw Item 8 and `10-Q` Item 1 text.

11. Updated the main document-layer chunking in `ingest_stock.py`, `ingest_macro.py`, and `ingest_knowledge.py` to use `SentenceSplitter(chunk_size=550, chunk_overlap=60)` for the vector indexes.

12. Optimized the query hot path by caching loaded indexes and glossary docs, reusing retrievers within a request, reducing stock retrieval fan-out, and only calling the graph layer for graph-heavy questions.

## 2026-03-28

13. Refactored the graph layer from per-ticker persisted graphs to one shared `PropertyGraphIndex` that keeps global glossary and macro nodes once, upserts each stock into the same graph with a manifest, and makes `query.py` read ticker-specific context from that shared graph.

14. Updated `ingest_stock.py` to enrich company profile and stock document metadata with Polygon ticker-overview sector data and top related tickers as peers, while keeping a safe SEC-based fallback when `POLYGON_API_KEY` is not configured.

15. Updated `ingest_stock.py` so 10-K refresh now checks local filing JSONs first, stores filing and next-release dates in metadata, and estimates the latest next 10-K date from fiscal year-end plus the company's historical filing pattern with filer-deadline support when detectable.

16. Updated `ingest_stock.py` so 10-Q refresh now checks local filing JSONs first, stores filing and next-release dates in metadata, and estimates the latest next 10-Q date from the next fiscal quarter-end plus the 40-day or 45-day SEC deadline while skipping Q4 by inferring `Q1/Q2/Q3` from the fiscal-year pattern when needed.

17. Updated `ingest_stock.py` so retained `10-K` and `10-Q` archives now behave like rolling windows by deduping filings by accession, keeping only the newest retained set, deleting dropped filing JSONs, and rebuilding stock docs and indexes from that exact queue.

18. Updated `ingest_graph.py` so the shared property graph now persists directly under `storage/graph` instead of `storage/graph/shared`, with a legacy-path migration that moves existing graph files into the new root location.

19. Updated `main.py` so the ticker prompt appears before the heavy `query` import chain by moving interactive analysis into `main()` and lazily importing `query` and `ingest_stock` only where they are needed.

20. Updated `ingest_stock.py`, `ingest_macro.py`, and `ingest_knowledge.py` to use `SentenceSplitter(chunk_size=800, chunk_overlap=70)`, and trimmed stock vector-index metadata so filing-derived docs no longer exceed the chunk limit and block stock index creation.

21. Updated `ingest_graph.py` to sanitize the shared property graph store on load by removing orphaned relations and triplets that reference missing nodes, which fixed the old shared-graph retrieval crash for `AAPL`.

## 2026-03-29

22. Updated `main.py` to time the analysis call and print the total answer-generation duration in seconds after the response is returned.

## 2026-03-30

23. Updated `main.py` to add a robust `top5-by-sector` bulk mode that screens S&P 500 companies by live Yahoo Finance market cap, groups them with the same SEC SIC-derived 9-sector mapping used in `ingest_stock.py`, refreshes only the stock SQL database for the selected names, and accepts quoted, uppercase, underscored, or spaced command variants instead of misreading them as tickers.

27. Updated `analysis.py` to support `plot <ticker>` benchmark charts from the existing stock SQL data, generating Seaborn revenue-trend and key-ratio comparison plots against same-sector peers already present in the database.

29. Updated `analysis.py` and `ingest_graph.py` so each SQL benchmark run now generates a deterministic summary and conclusion, upserts those analysis docs into the shared graph, and makes graph retrieval surface the stored benchmark analysis text for matching ticker queries.

30. Updated `analysis.py`, `ingest_stock.py`, and `main.py` so stock ingestion now runs the SQL benchmark analysis automatically without generating plots, stores the summary/conclusion in the shared graph, and reuses the same analysis path for the database-only bulk ingest flow.

## 2026-04-04

31. Updated `ingest_stock.py` and `ingest_graph.py` to replace the OpenAI model wiring with Gemini 2.5 Flash for text generation and graph extraction, replace OpenAI embeddings with `voyage-finance-2`, and fail fast with a clear message when the required Gemini or Voyage API key is missing.

32. Updated `ingest_stock.py` so Gemini and Voyage API key loading now accepts the exact `Gemini_API_KEY` / `Voyage_API_KEY` names and automatically strips a stray leading `=` from malformed `config.env` values, which restored successful Gemini and Voyage runtime validation.

33. Updated `query.py` and `ingest_stock.py` to detect legacy OpenAI-sized vector indexes after the Voyage switch, avoid forced re-embedding on the question path, and fall back to fresh SQL plus raw stock, macro, and glossary docs so ticker analysis still works under the current Voyage rate limits.

34. Simplified `query.py` by removing the raw stock, macro, and glossary migration fallback path, restoring rebuild-first index refresh for outdated indexes, and gating graph setup so it only runs for graph-style questions.

35. Simplified `ingest_stock.py` and `ingest_graph.py` by removing the old API-key compatibility shim, switching both files to direct required `Gemini_API_KEY` and `Voyage_API_KEY` config reads, and keeping the Gemini/Voyage model wiring unchanged.

36. Updated `query.py` so Voyage rate-limit failures during stock, knowledge, or macro index refresh and query-time retrieval now fail soft and fall back to SQL plus raw filing, macro, and glossary docs instead of crashing the analysis path.

37. Replaced the placeholder `ingest_news.py` with a NewsAPI-based broad-news pipeline that scores articles in Gemini batches, summarizes and extracts hybrid metadata per `Metadata Extractor.md`, rebuilds a dedicated news vector store, and upserts permanent high-score news fact docs into the shared graph through a new `ingest_graph.py` news-doc manifest path.

## 2026-04-06

38. Added `test.py` to fetch one general BBC business news item and print the metadata generated by `ingest_news.py`, and tightened the news company/ticker matcher so short ticker aliases no longer create bogus metadata hits like `A`, `AN`, or `IT`.

## 2026-04-07

39. Rewrote the main analysis prompt in `query.py` from the new `README.md` and the current system design so responses now focus on low-hallucination multi-angle diagnosis, cause attribution, macro regime effects, benchmark/news takeaways, and explicit fact-versus-inference reasoning from the app's own retrieved context.

40. Updated `query.py` and `main.py` so every stock answer now runs the SQL benchmark analysis first, folds a short analysis summary into the final prompt, returns the analysis artifacts alongside the answer, and opens the generated benchmark plots after the response is printed.

41. Updated `main.py` so benchmark plots are opened from absolute paths and only reported as opened on success, falls back to printing generated plot paths otherwise, and switches Windows to the selector event-loop policy before runtime initialization to reduce the shutdown-time SSL transport error after analysis finishes.

42. Updated `ingest_stock.py` and `query.py` so query-time stock index recovery now rebuilds the stock index without forcing a shared-graph refresh, which prevents ordinary analysis questions from hanging in graph insertion and triggering the repeated async SSL shutdown errors when the stock index is missing or stale.

43. Updated `analysis.py`, `query.py`, `main.py`, and `ingest_stock.py` so each ticker's SQL benchmark analysis now runs at most once per day and is then reused for later same-day calls while its cached summary still feeds every final answer, and expanded the SIC-derived sector mapping so auto manufacturers like `TSLA` resolve into the existing sector buckets and can generate daily benchmark plots.

44. Updated `query.py` to shorten the generated stock-analysis output by merging the old financial-diagnosis section with the filing, benchmark-analysis, and news takeaways into one compact section and explicitly instructing the model to keep each section high-signal and concise.

## 2026-04-16

45. Updated `ingest_stock.py`, `analysis.py`, `main.py`, `query.py`, and related metadata flow so the broad SIC-derived business bucket is now stored and described as `industry`, the specific SIC description is now stored and described as `sector`, and downstream benchmark, plot, graph, and prompt wording now follows that swapped classification consistently.

46. Corrected the classification swap across `ingest_stock.py`, `analysis.py`, `main.py`, `query.py`, and related flows so `industry` is now the broader SIC-derived bucket like `Technology` or `Consumer Defensive`, while `sector` is the narrower subcategory like `Electronic Computers` or `beverage`, and the benchmark and metadata layers now use those meanings consistently.

47. Replaced the raw SIC-description subcategory mapping with an explicit `industry -> sector` taxonomy in `ingest_stock.py` and propagated it through `analysis.py`, `main.py`, `query.py`, and `ingest_graph.py` so companies now resolve into broad industries like `Consumer Staples` and narrower sectors like `Food, Beverage & Tobacco` or `Technology Hardware & Equipment`.

## 2026-05-01

48. Reworked `ingest_news.py` around an Article-first news-ingestion MVP: removed the Event/event_id concept, kept `graph_facts` on Article metadata, normalized typo/null handling, added Pydantic schemas, dependency-injected store/client protocols, Prompt A scoring, Prompt B extraction, score-band policy logic, and a high-level batch ingestion pipeline.

49. Added a cosine-similarity negative-anchor filter to `ingest_news.py` that embeds predefined irrelevant-topic anchors, compares each incoming article description/title against the anchor matrix, and filters high-similarity hard negatives before Prompt A.

50. Updated `test.py` from a small ad hoc print script into a read-only news scoring runner that fetches recent articles, prints progress while running, groups results into `band_A`, `band_B`, `band_C` audit/drop output, and `band_f`, and writes readable files under `test/` with score, reason, description/summary, and full available content.

51. Switched the news fetch path from NewsAPI to World News API in both `ingest_news.py` and `test.py`, using `World_News_API_KEY`/`WORLD_NEWS_API_KEY`, source URLs for Bloomberg, Reuters, Financial Times, The Wall Street Journal, The Economist, and AP, World News `summary` as the report description, and World News `text` as the full content.

52. Tightened Prompt A's permanent-memory standard so `band_A` requires a durable, concrete market-relevant development, not merely a generally financial topic, podcast, interview, rumor, promotional item, or broad trend.

53. Centralized prompt text into the new `prompt.py` module and updated `ingest_news.py` and `query.py` to import Prompt A, Prompt B, legacy news prompts, and the main stock-analysis prompt from that single prompt source.

54. Updated Prompt A model wiring in `test.py` so Prompt A can run through `gpt-4.1-mini` by default via the OpenAI chat-completions endpoint, while preserving explicit Gemini model support through the existing LlamaIndex/Gemini path; also made `ingest_stock.env()` accept an optional LLM model override.

55. Added `test.py` command-line controls for Prompt A model selection, output directory, World News query text, and later Prompt A batch/timeout/retry behavior, so the read-only scoring run can be adjusted without editing code.

## 2026-05-02

56. Added the fast negative-anchor filter to the `test.py` read-only runner, including Voyage embedding calls, max cosine similarity against `ingest_news.NEGATIVE_TEXTS`, a configurable threshold, and a new `test/fast_filtered` report file showing matched anchor, similarity, description, and content.

57. Added OpenAI Prompt A timeout/retry handling and smaller default Prompt A batches in `test.py` so long World News content batches are less likely to fail the entire run on a transient timeout or dropped connection.

58. Added a terminal distribution summary before file generation in `test.py`, printing total fetched articles plus counts for `fast_filtered`, `band_A`, `band_B`, `band_C`, and `band_f` before writing the report files.

59. Extended Prompt A with policy-article detection in `prompt.py`, including allowed policy categories, policy status guidance, future-action scoring rules, and structured `is_policy_article` / `policy` output fields.

60. Added policy parsing and validation to `ingest_news.py` through `PolicySignal`, `is_policy_article`, and policy status/category normalization, and carried Prompt A policy fields into the merged payload passed to Prompt B and Article metadata.

61. Updated `test.py` report formatting to print Prompt A policy fields for scored articles and made that formatting backward-compatible so older score objects without policy attributes do not crash report generation.

62. Added public-market materiality, local/state policy caps, unresolved-project caps, and policy-status clarification language to Prompt A so important topics without concrete market transmission are held to lower bands.

63. Restored the score `2-4` behavior back to the original dropped-program path after the later band-C wording experiment: Prompt A again emits `drop`, `ingest_news.py` maps scores `2-4` to `drop`, and `test.py` keeps `band_C` only as the readable audit/report file for dropped articles.

64. Hardened Prompt A policy parsing in `ingest_news.py` so a plain-text `policy` value from the LLM is converted into a `PolicySignal.status_reason` instead of crashing report generation; policy presence now also marks the item as a policy article when the boolean flag is omitted.

65. Refactored the `test.py` read-only World News scoring flow into a shared function and added a band_A metadata inspection path that runs Prompt B only for band_A articles, prints each extracted Article metadata JSON block, and remains non-ingesting.

66. Changed the news filtering/scoring flow so the fast negative-anchor filter embeds only `title + first sentence of description`, Prompt A receives only `title + description`, and full article content remains reserved for later Prompt B metadata and graph-fact extraction on selected articles.

67. Switched the fast negative-anchor filter embedding path from Voyage to the local SentenceTransformers `all-MiniLM-L6-v2` model, added a reusable `SentenceTransformerEmbeddingClient`, and updated the test runner default/help text accordingly.
