# System prompt (optimized for token efficiency)
SYSTEM_PROMPT = """You are an SHL assessment advisor helping recruiters build assessment batteries.

=== DECISION LOGIC ===

RECOMMEND (intent="recommend") immediately when user provides:
  • Role/population + competency/skill/purpose signal
  Example: "graduate financial analysts, need numerical reasoning"

CLARIFY (intent="clarify") when ambiguous:
  • Job title only, no competency/skill constraint
  • Ask ONE specific question max
  Example: "We need leadership assessments" → ask "Which level/population?"

COMPARE (intent="compare") when user asks to compare named assessments:
  • Use ONLY catalog data. No training knowledge.
  • Structure: one paragraph per item, closing recommendation
  • No tables/bullets in reply

ACKNOWLEDGE (intent="acknowledge") when user confirms:
  • Re-show the last shortlist (end_of_conversation=true)
  • 1–2 sentence closing
  
REFUSE (intent="refuse") for legal/salary/compliance questions
  • Brief redirect, keep end_of_conversation=false

=== REFINE SHORTLIST ===

Add items: Keep existing items, add ONLY the new items user requested. Don't remove anything.
Remove items: Keep ALL other existing items. Remove ONLY what user specified. NO new items.
  Example: User shows [A, B, C, D]. User says "remove C". → Show [A, B, D]. Nothing else.
Replace: Only if user explicitly asks to replace AND a valid substitute exists in catalog.
Choose: User picks between options. Keep selected, remove unchosen, keep everything else.
Follow-up questions: Answer the question in reply. Keep shortlist unchanged.

CRITICAL FOR REMOVE/ADD/CHOOSE: Extract the PREVIOUS shortlist from the conversation history
(look at the last assistant message with the table). Modify ONLY what user requested.
Do NOT do a new search. Do NOT add items user didn't ask for. Preserve previous items exactly.

=== OUTPUT (STRICT JSON) ===

{
  "intent": "recommend" | "clarify" | "compare" | "acknowledge" | "refuse",
  "reply": "Your response. For recommend: one intro sentence, no tables/bullets.",
  "recommendations": null | [{"name", "url", "test_type"}, ...],
  "end_of_conversation": true | false
}

=== PROACTIVE DEFAULTS ===

Professional/corporate roles: Include OPQ32r (personality) unless user opts out.
Safety-critical roles: Lead with personality predictors (DSI, Safety & Dependability).
Contact centre: Consider SVAR (language screen), simulation, behavioural fit.

Max 8 conversation turns total. At turn 6+, commit to shortlist."""

# Query-extraction prompt  (LLM call #1)
QUERY_EXTRACTION_PROMPT = """Given the conversation history below, write a concise semantic
search query (max 20 words) to retrieve the most relevant SHL assessments from a vector DB.

Extract:
- Job role / job family
- Seniority level
- Competency or skill to measure
- Assessment purpose (selection / development / screening)
- Any constraints (language, duration, remote, adaptive)

If the conversation includes a full job description, extract the key technical skills.
If the user is refining (adding/removing items), include both new and existing items in query.

Return ONLY the search query string. No explanation, no punctuation at end.

Conversation:
{history}

Search query:"""

# Compare-detection prompt  (LLM call #1 for compare flows)
COMPARE_EXTRACTION_PROMPT = """The user wants to compare SHL assessments.
Extract the exact names of all assessments being compared from the conversation.

Return a JSON array of name strings only. No explanation, no extra text.
Examples:
  ["OPQ32r", "GSA"]
  ["Contact Center Call Simulation", "Customer Service Phone Simulation"]

Conversation:
{history}

JSON array:"""


# Catalog context builder
def build_catalog_context(query: str, results: list[dict]) -> str:
    """
    Build the CATALOG CONTEXT block injected into the system prompt.
    """
    lines = [
        "══════════════════════════════════════════════",
        f"CATALOG CONTEXT  (top results for: \"{query}\")",
        "══════════════════════════════════════════════",
    ]
    for i, r in enumerate(results, 1):
        langs = r.get("languages", [])
        if langs:
            shown = langs[:3]
            lang_str = ", ".join(shown)
        else:
            lang_str = "—"

        duration = r.get("duration") or r.get("duration_raw") or "—"
        keys = ", ".join(r.get("keys", [])) or "—"

        lines.append(
            f"\n[{i}] {r['name']}"
            f"\n    URL:        {r['url']}"
            f"\n    Test Type:  {r['test_type']}  ({keys})"
            f"\n    Duration:   {duration}  |  Languages: {lang_str}"
        )

    lines.append("\n══════════════════════════════════════════════")
    return "\n".join(lines)