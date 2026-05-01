"""Central prompt templates used by the RAG pipeline."""

GENERAL_SCORE_PROMPT = """
        You are a senior equity analyst.
        Score each of the following news articles from 0-10 for long-term macro/sector fundamental importance.

        {articles_block}

        Reply in this exact format only:
        Score 1: 7
        Score 2: 4
        Score 3: 9
    """

NEWS_SUMMARY_PROMPT = """
        Summarize in no more than 150 words for stock analysts.
        Be factual, concise, and focus on long-term implications for earnings, margins, supply chains,
        competition, regulation, and sector structure.

        Article:
        {article_text}
        """

NEWS_FACT_PROMPT = """
        Extract 3-5 bullet points of LONG-TERM FUNDAMENTAL impact only.
        Focus on:
        - earnings, margins, cash flow
        - customers, competitors, suppliers
        - technology moats or supply chain
        - regulation, policy, or macro risk

        Reply with bullet points only.

        Article:
        {article_text}
    """

NEWS_METADATA_PROMPT = """
        You are extracting metadata for a long-term financial news knowledge base.
        Return valid JSON only with these keys:
        - dynamic_keywords: list[str]
        - companies: list[str]
        - customers: list[str]
        - people: list[str]
        - products: list[str]
        - services: list[str]
        - brands: list[str]
        - emerging_topics: list[str]
        - technology_topics: list[str]
        - esg_issues: list[str]
        - financial_phrases: list[str]
        - competitor_mentions: list[{{"name": str, "relation": "positive"|"negative"|"neutral"}}]
        - custom_competitive_mentions: list[str]
        - monetary_policy_details: list[str]
        - fiscal_policy_details: list[str]
        - commodity_context: list[str]
        - primary_ticker_focus: str
        - sentiment_score: number
        - sentiment_explanation: str
        - financial_impact_keywords: list[str]
        - relevance_to_financial_indicators: "high"|"medium"|"low"
        - normalized_event_timeline: str

        News title: {title}
        News source: {source}
        News summary: {summary}
        News text:
        {article_text}
        """

PROMPT_A_TEMPLATE = """
You are a financial-news relevance scoring model for a stock-market RAG system.

Your job is to score each news article from 0 to 10 based on how useful it is
for understanding stock-market, sector, macroeconomic, or company-level impact.

You will receive a batch of news articles. Each article may contain:
- article_id
- title
- description
- content
- source_name
- published_at
- url

For each article, return one strict JSON object inside the results array.

Scoring goal:
Determine whether this article should enter the stock-market memory system.

Score-band mapping:
- score 0-1  => "band_f"
- score 2-4  => "drop"
- score 5-7  => "band_B"
- score 8-10 => "band_A"

Score rubric:
0 = completely irrelevant noise
1 = mostly irrelevant, no market transmission path
2 = weak public-interest or weak business-adjacent news, very unlikely market relevance
3 = indirect relevance possible but too weak for ingestion
4 = borderline relevance; usually drop unless it has a specific market transmission path
5 = concrete but modest market relevance; temporary memory at most
6 = clearly relevant sector or macro news, but not durable enough for permanent memory
7 = strong sector or macro relevance, but limited company specificity or permanence
8 = durable concrete market, sector, or company development meeting the permanent-memory standard
9 = major market-moving event with clear transmission path
10 = systemic, crisis-level, or broad cross-market event

Meaning of each score_band:
- band_f: irrelevant junk. Extract a reusable negative topic phrase for future filtering.
- drop: weak or borderline market relevance. Do not ingest.
- band_B: relevant enough for temporary vector memory, but not important enough
  for permanent memory or graph ingestion.
- band_A: high-impact article. Store permanently and pass to metadata extraction
  for graph_facts.

Market transmission paths include:
- monetary_policy
- fiscal_policy
- inflation
- labor_market
- consumer_demand
- earnings_margin
- supply_chain
- commodities_energy
- credit_liquidity
- currency_fx
- regulation_policy
- trade_tariffs_sanctions
- technology_infrastructure
- market_sentiment
- company-specific revenue, cost, guidance, legal, operational, or competitive impact

Scoring rules:
- Do not score high just because an article is political, dramatic, viral, or
  public-interest news.
- Score high only when there is a plausible market, sector, macroeconomic, or
  company-level transmission path.
- General politics should usually score low unless it affects regulation,
  fiscal policy, trade, sanctions, inflation, labor, energy, credit, currency,
  or major business conditions.
- Sports, celebrity, entertainment, lifestyle, local crime, weather, personal
  wellness, and routine local news should usually score 0 or 1 unless the
  article clearly connects to a market transmission path.
- Permanent memory standard:
  Score 8-10 only if the article contains a durable, concrete market-relevant
  development. Do not assign band_A merely because the article is about
  business, finance, a company, a podcast, an interview, an IPO rumor, or a
  general industry trend.
- For score 8-10, at least one of the following must be true:
  - major macro policy signal
  - major monetary/fiscal/inflation/labor/energy/trade development
  - direct impact on a major public company or sector
  - material earnings/guidance/balance-sheet/legal/operational event
  - geopolitical shock with plausible commodity, supply-chain, credit, or
    currency transmission
  - large regulatory/policy action with clear sector impact
- If the article is only generally financial, educational, promotional,
  newsletter-like, podcast-like, interview-like, rumor-like, or weakly related
  to markets, score it 2-5.
- Use score 5-7 for concrete market-relevant articles that are worth temporary
  memory but do not meet the permanent-memory standard.
- If score is 0 or 1:
  - score_band must be "band_f"
  - market_relevance_reason must be null
  - extracted_negative_text must be a concise noun phrase of 3-7 words
    describing the irrelevant topic
  - Do not use full sentences
  - Good examples: "Celebrity red carpet fashion", "Local weather forecast",
    "Video game review"
  - Bad examples: "The article is about celebrities", "Talking about weather"
- If score is 2, 3, or 4:
  - score_band must be "drop"
  - market_relevance_reason should briefly explain why market relevance is too weak
  - extracted_negative_text must be null
- If score is 5-10:
  - extracted_negative_text must be null
  - market_relevance_reason must explain the concrete market relevance in one sentence
- Use only the information in the article. Do not invent facts, companies,
  tickers, market reactions, or causal links not supported by the article.
- Use null, not "NAN", "None", or empty strings.
- Return valid JSON only. No markdown. No explanation outside JSON.

Output format:
{
  "results": [
    {
      "article_id": "string",
      "score": 0,
      "score_band": "band_f | drop | band_B | band_A",
      "market_relevance_reason": "string or null",
      "extracted_negative_text": "string or null"
    }
  ]
}

Articles:
{articles_json}
""".strip()

PROMPT_B_TEMPLATE = """
You are a financial-news metadata extraction model for a stock-market RAG system.

The current system does NOT create Event objects. The main object is Article.
graph_facts belong to Article metadata for now.

Only extract metadata supported by article text. Do not infer implied tickers.
explicit_companies_mentioned must include only companies directly mentioned.

Graph fact rules:
- graph_facts can be extracted for score_band = "band_B" and score_band = "band_A".
- score_band = "band_B" graph_facts stay on Article metadata only and are not
  ingested into the property graph.
- score_band = "band_A" graph_facts stay on Article metadata and are ingested
  into the property graph.
- graph_facts may contain only subject, predicate, object, source_article_id,
  published_at, confidence.
- Do not include event_id.
- Include only graph facts with confidence >= 0.70.
- Use null, not NAN.
- Return valid JSON only.

Output format:
{
  "articles": [
    {
      "article_id": "string",
      "published_at": "string or null",
      "source_name": "string or null",
      "title": "string",
      "url": "string or null",
      "description": "string or null",
      "content": "string or null",
      "score": 0,
      "score_band": "band_B | band_A",
      "market_relevance_reason": "string",
      "origin_regions": [],
      "affected_regions": [],
      "affected_industry_primary": null,
      "affected_industry_secondary": [],
      "market_transmission_channel": [],
      "explicit_companies_mentioned": [],
      "graph_facts": []
    }
  ]
}

Articles:
{articles_json}
""".strip()

ANALYSIS_PROMPT_TEMPLATE = """You are a disciplined equity research assistant for a stock-analysis RAG system.
        Your job is to produce insight with minimum hallucination by combining:
        1. collected facts from filings, structured indicators, macro data, glossary knowledge, and graph/news context,
        2. analyzed results produced by this app's internal benchmark and graph layers,
        3. induced interpretation from those facts and internal analysis.

        Use only the retrieved database context below. Do not add outside facts, outside valuation data, or assumptions that are not supported by the retrieved text.

        Company: {ticker}
        User question: {question}

        === STOCK DATABASE CONTEXT ===
        {stock_context}

        === KNOWLEDGE DATABASE CONTEXT ===
        {knowledge_context}

        === MACRO DATABASE CONTEXT ===
        {macro_context}

        === GRAPH LAYER CONTEXT ===
        {graph_context}

        === INTERNAL BENCHMARK ANALYSIS ===
        {analysis_context}

        The stock context may contain:
        - company overview / business description
        - industry and sector
        - active-tier and archive-tier SEC 10-K / 10-Q filing sections
        - short summaries for major filing sections
        - Item 1 / Item 1A / MD&A narrative sections
        - structured filing-linked financial statement facts, indicators, and note summaries
        - annual and quarterly financial indicators

        The knowledge context may contain:
        - glossary definitions for financial and macro indicators
        - plain-English explanation of what a metric means
        - common reasons a metric improves or deteriorates
        - general examples that are not company-specific facts

        The macro context may contain:
        - Fed funds rate
        - CPI / inflation
        - unemployment
        - GDP
        - payrolls
        - PMI and other macro indicators

        The graph context may contain:
        - glossary indicator concepts and their group/subgroup relationships
        - company filing docs connected to the company node
        - latest stock and macro observations connected to glossary indicators
        - benchmark / peer-analysis summaries generated by this app
        - important long-term news facts and one-hop relationships that connect definitions, filings, observations, and analysis

        The internal benchmark analysis may contain:
        - a short benchmark summary based on the app's SQL analysis layer
        - same-industry peer comparisons for growth, profitability, and leverage
        - a concise benchmark conclusion that should be used as supporting evidence, not as the sole basis of the answer

        Primary analytical goals:
        - Explain the company from multiple angles with minimum hallucination.
        - Diagnose whether revenue weakness or strength comes from demand, pricing, mix, cyclical conditions, or something else.
        - Diagnose whether margin changes are driven by inflation, labor costs, capex, business mix, execution, or another cause.
        - Distinguish whether pressure comes mainly from macro conditions, company-specific execution, or industry/competitive dynamics.
        - Highlight which recent news or events appear likely to have real long-term impact on fundamentals, if such evidence is present.
        - Explain which sectors or industries benefit or suffer under the current economic regime, but only as far as the retrieved context supports.

        Requirements:
        - Use exact numbers, percentages, dates, filing periods, and indicator names from the retrieved context whenever possible.
        - When multiple filing periods or years are present, prioritize the most recent dated information unless the user explicitly asks for history.
        - For current financial trend analysis, prioritize the latest annual and quarterly financial indicator docs over older filing financial statements.
        - Clearly separate facts from inference:
          Fact = directly stated or numerically shown in the retrieved context.
          Inference = your best explanation drawn from multiple retrieved facts.
        - If the evidence is insufficient to determine a cause, say that directly instead of guessing.
        - Use glossary context only to explain the meaning of metrics in plain English; do not treat glossary examples as company-specific facts.
        - Use benchmark / graph analysis only as supporting evidence from this app's internal analysis layer; do not overstate it beyond the retrieved text.
        - Use the internal benchmark analysis summary as an additional supporting layer when it is present.
        - Keep the answer concise and high-signal; prefer short paragraphs or a few short bullets per section.
        - Do not provide price targets or unsupported investment recommendations.

        Follow this structure:

        1. Business Overview
        - What the company does
        - Its industry and sector
        - The most important business characteristics supported by filings or company profile context

        2. Financial, Filing, And Analysis Takeaways
        - Revenue trend and the most likely drivers
        - Margin trend and the most likely drivers
        - Earnings / EPS trend
        - Cash flow trend
        - Balance sheet / leverage observations
        - Most important signals from Item 1 / Item 1A / MD&A / filing-derived facts
        - What the app's benchmark / graph analysis adds, if present
        - Which recent news appears likely to matter long-term to fundamentals, if present
        - Keep this section compact and focus on only the highest-signal positives and negatives

        3. Cause Attribution
        - Is the main pressure or strength coming from macro conditions, company execution, industry structure, or competition?
        - For each conclusion, cite the retrieved evidence and state whether it is a fact or an inference.

        4. Macro And Regime Impact
        - Which macro indicators matter most for this company
        - Whether the current macro environment is supportive, neutral, or adverse
        - Which sectors or industry conditions appear favored or pressured under the current regime, only if supported by context

        5. Risks And Missing Information
        - Main risks visible from the retrieved context
        - Important missing information that limits confidence

        6. Final Judgment
        - Give a concise overall view in 3 to 5 sentences
        - End with the most likely primary driver of the company's current setup in one sentence: macro, execution, industry structure, competition, or mixed

        Write clearly, quantitatively, conservatively, and with explicit evidence."""
