# Stock_RAG

Stock_RAG is an equity-research RAG project that I am building to move beyond basic document retrieval and toward **reasoned stock analysis**.

The end goal is to create a system that can look at a company from multiple angles at once:

- its long-term financial statements and operating trends
- the current macroeconomic environment
- what the business actually does based on SEC filings
- how it compares with peers and industry averages
- which news matters for the stock and which news is just noise
- how all of those pieces connect through a graph layer

In other words, this project is meant to become a **stock research assistant**, not just a chatbot over 10-Ks.

---

## What I want this project to do

Most stock-analysis tools can retrieve facts, but they often do a weak job answering the harder question:

**Why is this company performing the way it is right now?**

This project is designed to answer questions like:

- Is revenue weakness coming from demand destruction, pricing pressure, or a cyclical slowdown?
- Are margin changes driven by inflation, labor costs, capex, or business mix?
- Is the company under pressure because of macro conditions, company-specific execution, or industry competition?
- Which recent news is likely to have real long-term impact on fundamentals?
- Which sectors or industries benefit or suffer under the current economic regime?

The long-term vision is to make the model understand not just raw numbers, but also the **causal story behind the numbers**.

---

## System vision

I want the project to combine five layers of intelligence.

### 1. Company fundamentals layer
For each stock, the system should ingest and understand:

- annual and quarterly financial history
- core accounting line items
- derived financial ratios and performance indicators
- 10-K and 10-Q filing sections
- business description, risks, and management discussion
- filing recency and expected next filing dates

The point of this layer is to give the model a structured and historical view of the company.

### 2. Macro layer
The model should also understand the market environment around the company, including indicators such as:

- interest rates
- GDP
- inflation
- unemployment
- payroll data
- PMI and other cycle indicators

This layer should help the model reason about whether the environment is supportive, neutral, or adverse for the business.

### 3. Knowledge layer
Raw metrics are not enough. I want the system to know what the metrics mean.

For example, if net margin compresses or debt-to-equity rises, the model should understand:

- what that indicator represents
- what commonly causes it to improve or deteriorate
- how it should be interpreted in context
- when a ratio is informative versus misleading

So this project includes a glossary / concept layer that teaches the model the meaning behind financial and macro indicators.

### 4. News layer
I want the project to ingest broad market news and eventually stock-specific news, then score and classify it by long-term relevance.

The system should distinguish between:

- news that is probably short-lived noise
- news with real macro impact
- news that changes competitive position, margins, regulation, demand, supply chains, or capital allocation

This layer is especially important because some of the biggest market-moving events are not tied to a single ticker at first.

### 5. Graph layer
Finally, I want a graph layer that connects everything:

- companies
- filings
- financial indicators
- macro indicators
- knowledge concepts
- benchmark analysis
- high-value news facts

The graph should help the system understand relationships rather than just isolated chunks of text.

---

## What is already in the foundation

This repository already contains the core building blocks for that vision:

- stock ingestion
- macro ingestion
- knowledge / glossary ingestion
- graph construction
- news ingestion
- query-time retrieval and analysis
- benchmarking and peer comparison

So the project is no longer just an idea. The foundation exists; the next step is making it smarter, more accurate, and more useful.

---

## Current repository structure

### `ingest_stock.py`
Builds the company fundamentals layer.

This module is responsible for:

- pulling SEC-based financial data
- computing a large set of financial indicators
- storing structured records in SQLite
- keeping annual and quarterly history
- preserving 10-K and 10-Q filing context
- rebuilding the stock vector index
- attaching company/peer metadata used later in analysis

### `ingest_macro.py`
Builds the macro layer.

This module collects and stores key macro indicators so the model can analyze a company in the context of the broader economy instead of looking at the stock in isolation.

### `ingest_knowledge.py`
Builds the knowledge layer.

This part gathers glossary-style definitions and explanations for financial and macro concepts so the model can explain what metrics mean, not just repeat numbers.

### `ingest_news.py`
Builds the news layer.

This module is intended to ingest general market news across major buckets, score the long-term importance of articles, summarize them, extract structured metadata, and keep only the news that is worth retaining for stock research.

### `ingest_graph.py`
Builds the graph layer.

This layer connects the different data sources into a shared relationship graph so the system can reason over links between concepts, companies, observations, and events.

### `analysis.py`
Adds benchmark-style analysis.

This is where peer and sector comparison can become more analytical instead of just descriptive.

### `query.py`
Combines the layers at inference time.

This module retrieves the relevant company, macro, knowledge, and graph context and asks the model to produce a grounded equity-style answer.

### `main.py`
Acts as the entry point for the project.

It is the command-line surface where I can ingest data, run company analysis, and generate benchmark plots.

---

## Development roadmap

## Stage 1 — Build a reliable data foundation

The first goal is to make the data layer strong enough that every later answer has a trustworthy base.

This includes:

- collecting long-horizon annual and quarterly company fundamentals
- storing structured financial data in SQL
- building vector documents from the structured records and SEC filing content
- ingesting macroeconomic indicators regularly
- validating whether collected data is accurate and up to date
- reducing ingestion time so the pipeline can scale

A major concern here is not just ingestion, but **data quality control**.

---

## Stage 2 — Improve company and peer analysis

Once the data foundation is stable, the next step is making the analysis more comparative.

This includes:

- identifying sector and industry more reliably
- improving competitor and peer detection
- comparing company metrics with industry averages
- generating benchmark summaries and plots
- understanding whether a company is outperforming or lagging its group

The point of this stage is to move from isolated company summaries to **relative analysis**.

---

## Stage 3 — Make SEC filings truly useful

A lot of valuable information exists in filings, but it is not always directly structured.

I want the project to extract stronger signals from 10-K and 10-Q documents, such as:

- what the company really does
- business model details
- risk factors
- strategic priorities
- major operating drivers
- competitive positioning clues
- possible customer, supplier, or concentration signals

The goal is to stop treating filings as just searchable text and turn them into usable business intelligence.

---

## Stage 4 — Build smarter news understanding

General news is noisy, so I do not want to blindly dump all news into a vector store.

Instead, I want the system to:

- ingest broad weekly headlines from US, international, business, and technology categories
- score them for stock-market relevance
- extract structured metadata such as sectors, industries, policy, commodities, geography, and companies
- keep the high-value articles in retrieval and graph storage
- later connect broad news to affected sectors, industries, and individual companies

This stage is about turning news into a **fundamental signal layer**.

---

## Stage 5 — Build the graph-based reasoning layer

The graph layer is where the system starts to become more intelligent.

I want it to connect:

- indicators to their definitions
- companies to filings
- filings to business drivers and risks
- macro indicators to affected sectors
- benchmark analysis to peer groups
- important news to industries, companies, and macro themes

That should make it easier for the model to answer questions like:

- what changed
- why it changed
- what else is connected to that change
- whether the issue is likely company-specific, sector-wide, or macro-driven

---

## Stage 6 — Teach the model to reason over time series better

One open challenge in this project is that LLMs do not naturally reason perfectly over long numeric histories when the numbers are just dumped into text.

So I want to improve how time-series information is represented by:

- summarizing multi-year trends statistically
- highlighting important inflection points
- surfacing changes in growth, margins, leverage, and cash flow quality
- compressing long financial history into more meaningful analytical signals

This stage is about making the model better at seeing **trend, change, and regime shift** instead of just reading static tables.

---

## Stage 7 — Add macro-to-sector intelligence

Beyond single-company analysis, I want the project to understand broader economic regime effects.

Examples:

- which industries usually suffer when rates stay high
- which industries benefit from falling inflation
- how labor market strength affects consumer or cyclical businesses
- how energy, commodities, and policy shocks propagate through sectors

This is where the system starts to think less like a file retriever and more like a research analyst.

---

## Key design questions I am still working through

A few important questions are still open:

1. How should long financial history be compressed so the model can reason over it well?
2. How should I verify that model answers actually match the underlying data?
3. What is the best source for supplier risk, customer concentration, and competitor relationships?
4. How much news should stay in the vector store versus only in the graph layer?
5. What is the right balance between SQL analytics, vector retrieval, and graph retrieval?
6. How can I reduce ingestion latency without losing useful context?

These are not side questions. They are central to making the project actually useful.

---

## End goal

The final version of Stock_RAG should be able to do all of the following in one workflow:

- understand what a company does
- understand how the business has changed over time
- compare the company against peers and sector averages
- understand the current macro environment
- identify which news matters and why
- connect concepts through a graph instead of isolated retrieval
- explain the likely drivers behind performance, risk, and change

So the real goal of this repository is:

> build a stock research system that can move from **retrieval** to **reasoning**.

---

## Disclaimer

This project is for research and engineering exploration in retrieval-augmented financial analysis. It is not investment advice.
