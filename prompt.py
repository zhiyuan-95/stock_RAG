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
        You are generating graph-ingestion facts for a stock-market RAG system.

        Given the article text and extracted metadata, produce 3-5 bullet points only if the article contains market-relevant, fact-based information worth storing in the graph layer.

        Each bullet must be:
        - atomic: one factual claim per bullet
        - market-relevant: explain the transmission channel to stocks, sectors, commodities, rates, FX, or macro expectations
        - specific: include named entities, policy/event status, dates/timeframes, quantities, and affected sectors/tickers when available
        - grounded: do not add facts not supported by the article
        - non-duplicative: avoid repeating the same fact in different wording
        - neutral: distinguish confirmed facts from proposals, opinions, forecasts, rumors, or speculation

        Do NOT generate generic article summaries.
        Do NOT include background-only facts unless they directly explain the market impact.
        Do NOT generate bullets for articles with no new material development, even if the article discusses an important topic.
        Do NOT turn vague political promises, long-term debates, or unresolved discussions into confirmed market events.

        If the article does not contain at least 2 concrete market-relevant facts, return:
        GRAPH_FACTS: []
        REASON: "No concrete market-relevant development."

        Output format:

        GRAPH_FACTS:
        - [Entity] [action/event] [object/target] [status] [date/timeframe]; market relevance: [transmission channel]; affected areas: [sectors/tickers/assets if available]; certainty: [confirmed/proposed/uncertain/opinion].
        - ...
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
- source_name
- published_at
- url

For Prompt A, score only from the title and description fields. Full article
content is intentionally not provided at this stage. Do not assume facts that
are not supported by the title or description. Full content is used later only
for metadata and graph-fact extraction from articles classified as band_A.

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
10 = confirmed systemic, crisis-level, or broad cross-market event

Meaning of each score_band:
- band_f: irrelevant junk. Extract a reusable negative topic phrase for future filtering.
- drop: weak or borderline market relevance. Do not ingest.
- band_B: relevant enough for temporary vector memory, but not important enough
  for permanent memory or graph ingestion.
- band_A: high-impact article. Store permanently and pass to metadata extraction
  for graph_facts.

Core scoring gates:

    Do not confuse public importance with market relevance.
    A news article can be politically, legally, socially, or culturally important
    but still be irrelevant for this stock-market RAG system.

    Score 5 or above only when the article contains a specific investable market
    transmission path.

    A specific investable market transmission path means the article directly or
    strongly implies an effect on at least one of the following:
    - public companies or tickers
    - listed sectors, industries, or ETFs
    - commodities, energy, rates, FX, credit, or inflation
    - company revenue, costs, margins, guidance, balance sheet, legal risk, or operations
    - capital-markets activity such as IPO, debt refinancing, bankruptcy, bailout, M&A, or major financing
    - confirmed or concrete policy, regulation, tariff, sanction, export control, or central-bank action with economic impact

    Do not score 5 or above for vague, speculative, or second-order links such as:
    - could affect political sentiment
    - could shape future regulation someday
    - could influence public opinion
    - could matter if the situation escalates
    - could affect markets indirectly
    - could have long-term implications

    Band_B admission test:

    Before assigning score 5-7/band_B, the article must satisfy at least one:

    1. Public company / ticker / listed sector impact:
    The article names or clearly affects a public company, ticker, listed sector,
    ETF, commodity, rate, currency, credit market, or major investable industry.

    2. Macro or market variable impact:
    The article affects inflation, rates, labor, consumer demand, supply chains,
    commodities, energy, credit, FX, margins, revenue, or market liquidity.

    3. Concrete capital-markets event:
    The article reports IPO preparation, debt refinancing, bankruptcy risk,
    shutdown risk, bailout discussion, M&A, major financing, earnings, guidance,
    balance-sheet stress, or restructuring.

    4. Concrete policy/regulatory/geopolitical development:
    The article reports a proposed, threatened, announced, signed, implemented,
    blocked, delayed, reversed, or expired policy with a credible economic channel.

    5. Real operating impact:
    The article reports production cuts, plant shutdowns, capacity reduction,
    strikes, supply disruption, contract awards, major cost increases, demand shock,
    or infrastructure power/energy demand.

    If none of the Band_B admission conditions are satisfied, the article cannot
    receive score 5 or above.
    Assign score 2-4/drop if the article is weak public-interest, political, legal,
    policy-adjacent, geopolitical, local, or business-adjacent news.
    Assign score 0-1/band_f only if the article is a stable reusable junk category.

    Score caps:

    1. Private sports, entertainment, or cultural business:
    Capped at score 4 unless the article directly affects a public company, listed
    sector, media rights market, sports betting company, streaming platform,
    advertiser, or public-market asset.

    2. General politics:
    Election campaigns, endorsements, redistricting, voter rules, political
    profiles, and partisan disputes are capped at score 4 unless the article reports
    a concrete policy change, fiscal change, regulatory change, trade action, or
    market reaction.

    3. Legal/court stories:
    Court cases are capped at score 4 unless they directly affect a public company,
    industry regulation, product access, financial liability, merger approval,
    bankruptcy, labor rules, or material sector economics.

    4. Local/state infrastructure:
    Local or state infrastructure debates are capped at score 4 unless the article
    reports committed financing, construction start, contract award, cancellation,
    regulatory approval/rejection, or named public-company exposure.

    5. Geopolitical conflict:
    Military conflict, protests, or diplomatic tension are capped at score 4 unless
    the article directly affects oil, gas, shipping, commodities, supply chains,
    sanctions, defense spending, FX, credit risk, or named public companies.

    6. Private/nonprofit/local institution news:
    Private school, university, church, nonprofit, local administration, and local
    crime stories are capped at score 4 unless there is a clear public-company,
    sector, credit, or macroeconomic impact.

    Professional sports franchise ownership rule:

    A professional sports team sale, ownership transfer, franchise valuation, or league approval process is usually score 2-4/drop.

    Do not treat a private sports franchise as a public company.
    Do not call a sports team "public" unless the article explicitly says it is publicly traded or owned by a listed company.

    A sports franchise sale can receive score 5 or above only if the article directly names or clearly affects:
    - a publicly traded owner or buyer
    - a listed sports-betting company
    - a listed media, streaming, or broadcasting company
    - a major advertising or sponsorship market
    - a public credit/debt instrument
    - a listed sector or ETF

    Large private valuation alone is not enough for Band_B.

    Future-action rule:
    Future-action statements are capped at Band_B, not automatically Band_B.
    A future-action statement such as "will", "may", "plans to", "is considering",
    "threatens to", "vows to", or "signals" should be scored as follows:
    - score 2-4/drop if the article lacks a specific investable market transmission path
    - score 5-7/band_B if the article has a specific investable transmission path but the action is not final
    - score 8-10/band_A only if the action is officially announced with concrete terms, signed, implemented, blocked, delayed, reversed, or has already caused realized market impact

    Drop vs band_f rule:

    Use score 2-4/drop for weak market relevance, public-interest news, political
    news, legal news, geopolitical news, or policy-adjacent news that is not useful
    enough for stock-market memory.

    Use score 0-1/band_f only for stable reusable junk categories that are safely
    irrelevant across most future contexts.

    Do not use band_f for broad topics that can become market-relevant in another
    article, such as elections, court rulings, government policy, geopolitical
    conflict, sanctions, protests, labor disputes, infrastructure, immigration,
    public health, energy policy, military conflict, AI policy, or regulation.

    If these topics lack a concrete market transmission path, classify them as
    score 2-4/drop, not band_f.

    Band boundary correction rules:

    The most common classification error is over-upgrading public-interest articles
    into band_B. Avoid this error.

    Band_B is not for articles that are merely important, political, legal,
    dramatic, or business-adjacent. Band_B is only for articles that are useful for
    future stock-market retrieval.

    Before assigning score 5-7/band_B, the article must pass the Band_B admission
    test.

    Band_B admission test:
    At least one of the following must be true:
    1. The article names or clearly affects a public company, ticker, listed sector,
       ETF, commodity, rate, currency, credit market, or major investable industry.
    2. The article reports a concrete capital-markets event: IPO preparation, debt
       refinancing, bankruptcy risk, shutdown risk, bailout discussion, M&A, major
       financing, earnings, guidance, balance-sheet stress, or restructuring.
    3. The article reports a concrete policy/regulatory/geopolitical development
       with a credible economic channel: tariffs, sanctions, export controls,
       central-bank action, fiscal policy, energy policy, labor policy, financial
       regulation, antitrust, healthcare regulation, defense spending, or industrial
       policy.
    4. The article reports a concrete operating event: production cut, plant
       shutdown, capacity reduction, strike, supply disruption, contract award,
       major cost increase, demand shock, or infrastructure power/energy demand.
    5. The article directly affects inflation, rates, labor, consumer demand,
       supply chains, commodities, energy, credit, FX, margins, revenue, or market
       liquidity.

    If none of these conditions are met, the article cannot be score 5-7/band_B.
    It must be score 2-4/drop unless it is stable irrelevant junk.

    Weak second-order reasoning is not enough for Band_B.
    Do not assign Band_B based only on phrases such as:
    - could affect political sentiment
    - could shape future regulation
    - may have long-term implications
    - could affect public opinion
    - could matter if the issue escalates
    - might influence markets indirectly

    Band_F safety rule:
    Use score 0-1/band_f only for stable reusable junk categories such as sports
    game recaps, celebrity gossip, entertainment reviews, lifestyle advice, local
    crime, weather, animal rescue, obituaries, horoscopes, and routine cultural
    events.

    Do not use band_f for broad topics that can be market-relevant in another
    context, including elections, court rulings, government policy, geopolitical
    conflict, military conflict, protests, labor disputes, infrastructure,
    immigration, public health, AI policy, energy policy, or regulation.
    If these topics lack a concrete market transmission path, classify them as
    score 2-4/drop.

Future-action correction:
Future-action statements are capped at Band_B, not automatically Band_B.
A statement using "will", "may", "plans to", "is considering", "threatens to",
"vows to", or "signals" should be score 2-4/drop if it does not pass the Band_B
admission test.

Market materiality and unresolved-project rules:
1. Public-market materiality gate:
Do not assign band_A only because an article involves a large dollar amount,
government project, infrastructure plan, public works project, environmental dispute,
local/state policy, or public-interest issue.

For score 8-10, the article must report at least one concrete public-market
transmission path:
- direct effect on a public company’s revenue, costs, guidance, balance sheet,
  legal risk, or operations;
- direct effect on a broad listed sector, ETF, commodity, interest rate, currency,
  or credit market;
- confirmed national or cross-border policy action with clear market impact;
- realized market reaction reported in the article;
- confirmed funding, implementation, contract award, cancellation, or regulatory
  decision that materially affects investable companies or sectors.

If no concrete public-market transmission path is present, score must be 0-5.

2. Local/state policy cap:
State, local, municipal, regional, or project-specific policy articles are usually
score 2-5 unless they clearly affect a major public company, listed sector, ETF,
commodity market, national macro condition, or major public contract awarded to
publicly traded companies.

3. Unresolved project cap:
If an article is about a proposed, debated, uncertain, unfunded, legally challenged,
or not-yet-committed project, score it below 8 unless the article reports clear
realized market impact or direct material impact on public companies/sectors.

If financing is not secured, construction is not committed, or key approvals remain
unresolved, the maximum score is usually 5 or 6.

4. Policy status clarification:
Do not classify a project as "announced" merely because the article reports a
planning milestone, regulatory review, hearing, debate, environmental review, or
partial procedural approval.

Use "proposed" or "unclear" when the project remains unresolved.
Use "implemented" only when construction, enforcement, or policy execution has begun.

Output format:
{
  "results": [
    {
      "article_id": "string",
      "score": 0,
      "score_band": "band_f | drop | band_B | band_A",
      "market_relevance_reason": "string or null",
      "extracted_negative_text": "string or null",
      "is_policy_article": false,
      "policy": null
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
Carry is_policy_article and policy forward exactly from Prompt A. Do not invent
policy fields in Prompt B.

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
      "is_policy_article": false,
      "policy": null,
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
