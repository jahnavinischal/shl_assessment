import json
import os
import re
import logging
from difflib import SequenceMatcher

from groq import Groq

from catalog import search
from models import Message, Recommendation
from prompts import (
    SYSTEM_PROMPT,
    QUERY_EXTRACTION_PROMPT,
    COMPARE_EXTRACTION_PROMPT,
    build_catalog_context,
)

logger = logging.getLogger(__name__)

# Groq client  (lazy-initialised singleton)

_groq_client: Groq | None = None


def get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com/keys"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _llm(prompt: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    """Single Groq completion call. Returns raw text content."""
    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _llm_with_system(system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    """
    Two-message Groq call with explicit system + user separation.
    Keeps the large system prompt out of the user message token count,
    and allows Groq to cache the system prompt across calls.
    """
    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# Helpers

def _format_history(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        role = "User" if m.role == "user" else "Agent"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def _count_turns(messages: list[Message]) -> int:
    return sum(1 for m in messages if m.role == "user")


def _is_compare_query(messages: list[Message]) -> bool:
    """
    Quick heuristic check: does the latest user message look like a comparison?
    Avoids an LLM call for the common non-compare case.
    """
    last_user = ""
    for m in reversed(messages):
        if m.role == "user":
            last_user = m.content.lower()
            break

    compare_signals = [
        "difference between", "differ from", "compare", "vs ", "versus",
        "same as", "better than", "how does", "what is the diff",
        "is there a difference", "distinguish",
    ]
    return any(sig in last_user for sig in compare_signals)


def _is_refine_command(messages: list[Message]) -> bool:
    """
    Detect if the latest user message is a refinement command:
    'remove X', 'add Y', 'drop Z', 'keep X', 'replace X with Y', etc.
    
    Refinement commands modify the existing shortlist without starting a new search.
    """
    last_user = ""
    for m in reversed(messages):
        if m.role == "user":
            last_user = m.content.lower().strip()
            break
    
    # Short command patterns that modify existing list
    refine_signals = [
        "remove ", "drop ", "exclude ", "delete ",
        "add ", "include ", "replace ", "swap ",
        "keep ", "confirmed", "perfect", "good",
        "yes,", "no,",
    ]
    
    # Only treat as refinement if:
    # 1. Message is short (< 20 words) AND
    # 2. Contains a refine signal
    words = last_user.split()
    return len(words) <= 20 and any(sig in last_user for sig in refine_signals)


def _extract_search_query(messages: list[Message]) -> str:
    """LLM Call #1a: extract a FAISS search query from conversation history."""
    history = _format_history(messages)
    prompt = QUERY_EXTRACTION_PROMPT.format(history=history)
    try:
        query = _llm(prompt, max_tokens=60, temperature=0.1).strip().strip('"').strip("'")
        logger.info(f"[agent] Search query: {query!r}")
        return query
    except Exception as e:
        logger.warning(f"[agent] Query extraction failed: {e}")
        for m in reversed(messages):
            if m.role == "user":
                return m.content
        return "SHL assessment"


def _extract_compare_names(messages: list[Message]) -> list[str]:
    """
    LLM Call #1b: extract the names of assessments the user wants to compare.
    Returns a list of name strings (may be short/partial names).
    """
    history = _format_history(messages)
    prompt = COMPARE_EXTRACTION_PROMPT.format(history=history)
    try:
        raw = _llm(prompt, max_tokens=100, temperature=0.1).strip()
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip().strip("`")
        names = json.loads(raw)
        if isinstance(names, list):
            logger.info(f"[agent] Compare names extracted: {names}")
            return [str(n) for n in names if n]
    except Exception as e:
        logger.warning(f"[agent] Compare name extraction failed: {e}")
    return []


def _build_compare_context(names: list[str]) -> tuple[str, list[dict]]:
    """
    Build catalog context for a comparison query.
    Does a per-name targeted search (top-3 each) so both items appear in context,
    then a broad combined search for additional context.
    Returns (context_str, combined_results).
    """
    all_results: list[dict] = []
    seen_names: set[str] = set()

    # Per-name search ensures each named assessment appears near the top
    for name in names:
        for r in search(name, k=2):
            if r["name"] not in seen_names:
                all_results.append(r)
                seen_names.add(r["name"])

    # Broad combined search for supporting context
    if names:
        for r in search(" ".join(names), k=6):
            if r["name"] not in seen_names:
                all_results.append(r)
                seen_names.add(r["name"])

    query_label = " vs ".join(names) if names else "assessment comparison"
    context = build_catalog_context(query_label, all_results[:20])
    return context, all_results


def _safe_parse_json(raw: str) -> dict:
    """
    Parse LLM output as JSON, handling markdown fences and stray text.
    Returns a safe fallback dict on failure.
    """
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip().strip("`")
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[agent] JSON parse failed: {e}\nRaw (first 800):\n{raw[:800]}")
        return {
            "intent": "clarify",
            "reply": "Could you tell me more about the role you're hiring for?",
            "recommendations": None,
            "end_of_conversation": False,
        }


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _resolve_recommendations(
    recommend_names: list[str],
    catalog_results: list[dict],
) -> list[Recommendation]:
    """
    Fuzzy-match LLM-returned names back to catalog items.
    Includes full metadata (keys, duration, languages). Never invents items.
    """
    resolved = []
    used: set[str] = set()

    for name in recommend_names:
        if not name or not name.strip():
            continue
        best_item, best_score = None, 0.0
        for item in catalog_results:
            sim = _similarity(name, item["name"])
            if sim > best_score:
                best_score = sim
                best_item = item
        if best_item and best_score >= 0.55 and best_item["name"] not in used:
            resolved.append(Recommendation(
                name=best_item["name"],
                url=best_item["url"],
                test_type=best_item["test_type"],
                keys=best_item.get("keys") or [],
                duration=best_item.get("duration") or best_item.get("duration_raw") or None,
                languages=best_item.get("languages") or [],
            ))
            used.add(best_item["name"])

    return resolved[:10]



def _build_main_prompt(
    messages: list[Message],
    catalog_context: str,
    turn_count: int,
) -> tuple[str, str]:
    """Returns (system_prompt, user_message) tuple for _llm_with_system."""
    history = _format_history(messages)
    turn_note = ""
    if turn_count >= 3:
        turns_left = max(0, 4 - turn_count)
        if turns_left <= 1:
            turn_note = (
                "\n  TURN BUDGET: Near turn limit. "
                "If you have enough context, produce a recommendation NOW. "
                "Do not ask more clarifying questions.\n"
            )
    system = f"{SYSTEM_PROMPT}\n\n{catalog_context}"
    user = (
        f"{turn_note}"
        f"CONVERSATION HISTORY:\n{history}\n\n"
        f"Respond with a single valid JSON object (no fences, no extra text):"
    )
    return system, user



def _extract_last_shortlist_from_history(
    messages: list[Message],
    catalog_results: list[dict],
) -> list[Recommendation] | None:
    """
    Parse the last assistant message that contained a markdown table and
    extract the assessment names from it, then resolve them against the catalog.
    Used by the acknowledge intent to re-show the last shortlist.
    """
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        # Look for lines that are table rows: | 1 | Name | ...
        rows = re.findall(r"\|\s*\d+\s*\|\s*([^|]+)\|", msg.content)
        if rows:
            names = [r.strip() for r in rows if r.strip()]
            if names:
                resolved = _resolve_recommendations(names, catalog_results)
                if resolved:
                    return resolved
    return None


# Main agent entry point

def run_agent(messages: list[Message]) -> dict:
    """
    Process a full conversation history and return the next agent turn.

    Returns a dict matching ChatResponse:
      { "reply": str, "recommendations": list | None, "end_of_conversation": bool }
    """
    turn_count = _count_turns(messages)
    logger.info(f"[agent] Turn {turn_count}, {len(messages)} messages")

    # Step 1: Detect query type (cheap heuristics first)
    is_compare = _is_compare_query(messages)
    is_refine = _is_refine_command(messages) and turn_count > 1

    # Step 2: Build catalog context
    if is_compare:
        # For comparisons: extract the specific names being compared,
        # then build a targeted context that always includes those items.
        compare_names = _extract_compare_names(messages)
        catalog_context, catalog_results = _build_compare_context(compare_names)
        
    elif is_refine:
        # For refinement (remove/add/drop items): focus only on the previous shortlist.
        # This prevents the LLM from accidentally adding new items from search results.
        logger.info("[agent] Refinement command detected. Using previous shortlist context only.")
        prev_recs = _extract_last_shortlist_from_history(messages, [])  # Temp empty catalog
        
        if prev_recs:
            # Build catalog results from the previous recommendations
            catalog_results = [
                {
                    "name": r.name,
                    "url": r.url,
                    "test_type": r.test_type,
                    "keys": r.keys,
                    "duration": r.duration,
                    "languages": r.languages,
                }
                for r in prev_recs
            ]
            catalog_context = build_catalog_context(
                "previous shortlist (refining)",
                catalog_results
            )
        else:
            # Fallback if extraction fails
            search_query = _extract_search_query(messages)
            catalog_results = search(search_query, k=15)
            catalog_context = build_catalog_context(search_query, catalog_results)
    else:
        # For all other intents: semantic search on conversation summary
        search_query = _extract_search_query(messages)
        catalog_results = search(search_query, k=6)
        catalog_context = build_catalog_context(search_query, catalog_results)

    # Step 3: Call main LLM
    system_prompt, user_msg = _build_main_prompt(messages, catalog_context, turn_count)

    try:
        raw_output = _llm_with_system(system_prompt, user_msg, max_tokens=1500, temperature=0.2)
        logger.debug(f"[agent] Raw output:\n{raw_output[:600]}")
    except Exception as e:
        logger.error(f"[agent] Main LLM call failed: {e}")
        return {
            "reply": "I'm having trouble connecting. Please try again in a moment.",
            "recommendations": None,
            "end_of_conversation": False,
        }

    # Step 4: Parse JSON output
    parsed = _safe_parse_json(raw_output)

    intent = parsed.get("intent", "clarify")
    reply = parsed.get("reply", "")
    end_of_conversation = bool(parsed.get("end_of_conversation", False))
    raw_recs = parsed.get("recommendations")

    # Strip any stray markdown table the LLM snuck into the reply
    reply = re.sub(r"\n\|.*", "", reply, flags=re.DOTALL).strip()

    # Step 5: Handle each intent
    final_recommendations: list[Recommendation] | None = None

    if intent == "recommend" and isinstance(raw_recs, list) and raw_recs:
        # Resolve LLM names → catalog items via fuzzy matching
        names = [r.get("name", "") if isinstance(r, dict) else str(r) for r in raw_recs]
        final_recommendations = _resolve_recommendations(names, catalog_results)

        if not final_recommendations:
            # Fallback: fuzzy matching failed — use top FAISS results directly
            logger.warning("[agent] Fuzzy matching yielded nothing. Using top FAISS results.")
            final_recommendations = [
                Recommendation(
                    name=r["name"],
                    url=r["url"],
                    test_type=r["test_type"],
                    keys=r.get("keys") or [],
                    duration=r.get("duration") or r.get("duration_raw") or None,
                    languages=r.get("languages") or [],
                )
                for r in catalog_results[:10]
            ]

        # reply stays as the LLM's plain intro text

    elif intent == "compare":
        # Compare: reply is prose from the LLM grounded in catalog context.
        # No recommendations array. end_of_conversation stays false.
        final_recommendations = None
        end_of_conversation = False

    elif intent == "acknowledge":
        # User confirmed — close the conversation.
        # Re-show the last shortlist from conversation history (if any).
        # The LLM may have included recs in the acknowledge response; use them.
        # If not, extract from the last recommend turn in history.
        if isinstance(raw_recs, list) and raw_recs:
            names = [r.get("name", "") if isinstance(r, dict) else str(r) for r in raw_recs]
            final_recommendations = _resolve_recommendations(names, catalog_results)
        else:
            final_recommendations = _extract_last_shortlist_from_history(messages, catalog_results)

        end_of_conversation = True

    elif intent in ("clarify", "refuse"):
        # Nothing extra to do — reply and null recs are correct.
        final_recommendations = None
        end_of_conversation = False

    else:
        # Unknown intent — treat as clarify
        logger.warning(f"[agent] Unknown intent '{intent}', treating as clarify")
        final_recommendations = None
        end_of_conversation = False

    logger.info(
        f"[agent] intent={intent}, recs={len(final_recommendations) if final_recommendations else 0}, "
        f"eoc={end_of_conversation}"
    )

    return {
        "reply": reply,
        "recommendations": final_recommendations,
        "end_of_conversation": end_of_conversation,
    }