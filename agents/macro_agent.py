"""
Macro & Market Environment Agent (§4.3).

Single global instance — not per-stock.
Uses claude-sonnet-4-6 (strategic model) for nuanced economic interpretation.

Outputs per-sector macro modifiers (11 sectors), market_regime string,
and sector_ratings (overweight / market_weight / underweight + rationale).
"""

from __future__ import annotations

import json
from typing import Optional

import anthropic

from config import STRATEGIC_MODEL, MAX_TOKENS_STRATEGIC, ANTHROPIC_API_KEY, ALL_SECTORS

_PROMPT_TEMPLATE = """\
You are a macro economist and market strategist maintaining an ongoing
macro environment brief for US equity investors.

PRIOR REPORT (from {prior_date}):
{prior_report}
[If "NONE — initial report", produce a fresh brief from the data below.]

CURRENT MACRO DATA:
- Fed Funds Rate: {fed_funds}%
- 2yr Yield: {t2yr}% | 10yr Yield: {t10yr}% | Curve: {curve_slope}bps
- VIX: {vix}
- SPY 30d return: {spy_30d}% | QQQ 30d: {qqq_30d}% | IWM 30d: {iwm_30d}%
- WTI Oil: ${oil}/bbl | Gold: ${gold}/oz | Copper: ${copper}/lb
- Latest CPI: {cpi_yoy}% YoY | Unemployment: {unemployment}%
- Next scheduled releases: {upcoming_releases}

THESIS DRAFTING PROTOCOL — FOLLOW IN ORDER:
1. START WITH EXISTING: The prior report is your baseline. Preserve
   the macro thesis and supporting evidence that remains valid.
2. ADD NEW FINDINGS: Note any material changes in rate expectations,
   yield moves >10bps, VIX regime shifts, significant commodity moves,
   or new economic data releases since the prior report.
3. REMOVE STALE CONTENT: Remove commentary about conditions that have
   since changed (e.g., prior VIX spike that has resolved, rate
   expectations that have been revised). When in doubt, retain.

Keep the report to approximately 300 words.
End with a JSON block:
{{
  "market_regime": "<bull|neutral|caution|bear>",
  "key_theme": "<one sentence macro thesis>",
  "material_changes": <true|false>,
  "sector_modifiers": {{
    "Technology": <0.70-1.30>,
    "Healthcare": <0.70-1.30>,
    "Financial": <0.70-1.30>,
    "Consumer Discretionary": <0.70-1.30>,
    "Consumer Staples": <0.70-1.30>,
    "Energy": <0.70-1.30>,
    "Industrials": <0.70-1.30>,
    "Materials": <0.70-1.30>,
    "Real Estate": <0.70-1.30>,
    "Utilities": <0.70-1.30>,
    "Communication Services": <0.70-1.30>
  }},
  "sector_ratings": {{
    "Technology":             {{"rating": "<overweight|market_weight|underweight>", "rationale": "<1 sentence>"}},
    "Healthcare":             {{"rating": "<overweight|market_weight|underweight>", "rationale": "<1 sentence>"}},
    "Financial":              {{"rating": "<overweight|market_weight|underweight>", "rationale": "<1 sentence>"}},
    "Consumer Discretionary": {{"rating": "<overweight|market_weight|underweight>", "rationale": "<1 sentence>"}},
    "Consumer Staples":       {{"rating": "<overweight|market_weight|underweight>", "rationale": "<1 sentence>"}},
    "Energy":                 {{"rating": "<overweight|market_weight|underweight>", "rationale": "<1 sentence>"}},
    "Industrials":            {{"rating": "<overweight|market_weight|underweight>", "rationale": "<1 sentence>"}},
    "Materials":              {{"rating": "<overweight|market_weight|underweight>", "rationale": "<1 sentence>"}},
    "Real Estate":            {{"rating": "<overweight|market_weight|underweight>", "rationale": "<1 sentence>"}},
    "Utilities":              {{"rating": "<overweight|market_weight|underweight>", "rationale": "<1 sentence>"}},
    "Communication Services": {{"rating": "<overweight|market_weight|underweight>", "rationale": "<1 sentence>"}}
  }}
}}

Guidance for sector_modifiers:
- Assign each sector a modifier from 0.70 (strong macro headwind) to 1.30 (strong tailwind).
- Rising rates: headwind for Real Estate/Utilities, tailwind for Financials.
- High VIX: broadly negative but less severe for Consumer Staples/Healthcare (defensive).
- High oil: benefits Energy; hurts Consumer Discretionary/Industrials (input cost).
- Rate-cutting cycle + low VIX: benefits rate-sensitive growth sectors (Technology, Real Estate).
- Modifiers should vary across sectors — avoid assigning the same value to all sectors.
- market_regime drives RSI threshold selection in the technical scoring engine.

Guidance for sector_ratings:
- overweight: modifier >= 1.08, or sector has specific structural catalyst beyond the modifier.
- underweight: modifier <= 0.92, or sector faces specific structural headwind.
- market_weight: everything in between, no decisive tilt.
- rationale must name the specific macro mechanism (e.g., "Rising 10yr yield pressures cap
  rates and refinancing costs" not just "rising rates are bad for real estate").
- sector_ratings are the human-readable allocation signal consumed by the executor.

Output the full refreshed report followed by the JSON block.
"""

_DEFAULT_SECTOR_MODIFIERS = {s: 1.0 for s in ALL_SECTORS}

_VALID_RATINGS = {"overweight", "market_weight", "underweight"}
_OW_THRESHOLD = 1.08   # modifier >= this → overweight
_UW_THRESHOLD = 0.92   # modifier <= this → underweight


def _find_json_block(text: str, key: str = '"market_regime"') -> tuple[int, int]:
    """
    Find the start and end indices of the JSON object containing `key`.
    Uses balanced-brace scanning — handles nested dicts correctly.
    Returns (start, end) inclusive, or (-1, -1) if not found.
    """
    key_pos = text.find(key)
    if key_pos == -1:
        return -1, -1
    brace_pos = text.rfind('{', 0, key_pos)
    if brace_pos == -1:
        return -1, -1
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[brace_pos:], brace_pos):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return brace_pos, i
    return -1, -1


def _derive_sector_ratings(sector_modifiers: dict[str, float]) -> dict[str, dict]:
    """Fallback: derive OW/MW/UW ratings from modifier values when agent JSON is missing."""
    ratings = {}
    for sector, mod in sector_modifiers.items():
        if mod >= _OW_THRESHOLD:
            rating = "overweight"
            rationale = f"Macro tailwind (modifier {mod:.2f})"
        elif mod <= _UW_THRESHOLD:
            rating = "underweight"
            rationale = f"Macro headwind (modifier {mod:.2f})"
        else:
            rating = "market_weight"
            rationale = f"Neutral macro backdrop (modifier {mod:.2f})"
        ratings[sector] = {"rating": rating, "rationale": rationale}
    return ratings


def _fmt(val, fmt=".1f", default="N/A") -> str:
    if val is None:
        return default
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return default


def _extract_macro_json(text: str) -> dict:
    """Extract the trailing JSON block with sector_modifiers and sector_ratings."""
    start, end = _find_json_block(text)
    match_text = text[start:end + 1] if start != -1 else None

    if match_text:
        try:
            data = json.loads(match_text)
            # Validate and clamp sector modifiers
            mods = data.get("sector_modifiers", {})
            for sector in ALL_SECTORS:
                val = mods.get(sector, 1.0)
                mods[sector] = round(max(0.70, min(1.30, float(val))), 3)
            data["sector_modifiers"] = mods
            # Ensure market_regime is valid
            valid_regimes = {"bull", "neutral", "caution", "bear"}
            if data.get("market_regime") not in valid_regimes:
                data["market_regime"] = "neutral"
            # Validate sector_ratings; fall back to derived values for any missing/invalid entry
            raw_ratings = data.get("sector_ratings", {})
            validated_ratings = {}
            for sector in ALL_SECTORS:
                entry = raw_ratings.get(sector, {})
                rating = entry.get("rating", "")
                rationale = entry.get("rationale", "")
                if rating not in _VALID_RATINGS:
                    # Derive from modifier as fallback
                    mod = mods.get(sector, 1.0)
                    fallback = _derive_sector_ratings({sector: mod})[sector]
                    validated_ratings[sector] = fallback
                else:
                    validated_ratings[sector] = {"rating": rating, "rationale": rationale}
            data["sector_ratings"] = validated_ratings
            return data
        except (json.JSONDecodeError, ValueError):
            pass
    defaults = {
        "market_regime": "neutral",
        "sector_modifiers": _DEFAULT_SECTOR_MODIFIERS.copy(),
        "key_theme": "Macro data unavailable.",
        "material_changes": False,
    }
    defaults["sector_ratings"] = _derive_sector_ratings(defaults["sector_modifiers"])
    return defaults


def run_macro_agent(
    prior_report: Optional[str],
    prior_date: str,
    macro_data: dict,
    api_key: Optional[str] = None,
) -> dict:
    """
    Run the Macro & Market Environment Agent.

    Returns dict with:
      report_md: str
      macro_json: dict (with market_regime and sector_modifiers)
      market_regime: str
      sector_modifiers: dict[str, float]
    """
    client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    prior_text = prior_report or "NONE — initial report"

    prompt = _PROMPT_TEMPLATE.format(
        prior_date=prior_date,
        prior_report=prior_text,
        fed_funds=_fmt(macro_data.get("fed_funds_rate")),
        t2yr=_fmt(macro_data.get("treasury_2yr")),
        t10yr=_fmt(macro_data.get("treasury_10yr")),
        curve_slope=_fmt(macro_data.get("yield_curve_slope"), ".0f"),
        vix=_fmt(macro_data.get("vix")),
        spy_30d=_fmt(macro_data.get("sp500_30d_return")),
        qqq_30d=_fmt(macro_data.get("qqq_30d_return")),
        iwm_30d=_fmt(macro_data.get("iwm_30d_return")),
        oil=_fmt(macro_data.get("oil_wti"), ".2f"),
        gold=_fmt(macro_data.get("gold"), ".0f"),
        copper=_fmt(macro_data.get("copper"), ".2f"),
        cpi_yoy=_fmt(macro_data.get("cpi_yoy")),
        unemployment=_fmt(macro_data.get("unemployment")),
        upcoming_releases="See FRED calendar for next CPI/FOMC dates.",
    )

    response = client.messages.create(
        model=STRATEGIC_MODEL,
        max_tokens=MAX_TOKENS_STRATEGIC,
        messages=[{"role": "user", "content": prompt}],
    )

    full_text = response.content[0].text
    macro_json = _extract_macro_json(full_text)

    # Strip JSON block from markdown using the same balanced-brace scanner
    _start, _end = _find_json_block(full_text)
    if _start != -1:
        report_md = (full_text[:_start] + full_text[_end + 1:]).strip()
    else:
        report_md = full_text.strip()

    return {
        "report_md": report_md,
        "macro_json": macro_json,
        "market_regime": macro_json.get("market_regime", "neutral"),
        "sector_modifiers": macro_json.get("sector_modifiers", _DEFAULT_SECTOR_MODIFIERS.copy()),
        "sector_ratings": macro_json.get("sector_ratings", _derive_sector_ratings(_DEFAULT_SECTOR_MODIFIERS)),
        "material_changes": bool(macro_json.get("material_changes", False)),
    }
