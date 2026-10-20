"""Choose a battle tool from the user's intent, independently of retrieval hits."""

from __future__ import annotations

import re


def route_battle_question(question: str) -> str:
    """Default to information lookup; team context alone never requests strategy.

    Explicit decisions and matchup requests take precedence over definitions,
    including combined questions such as 'what is Sitrus Berry and should I use it?'.
    """
    text = " ".join(str(question or "").lower().split())
    decision_patterns = (
        r"\b(?:should|would|could) (?:i|we|my|our)\b",
        r"\bwhat (?:do|can) i (?:lead|bring|use|pick|choose)\b",
        r"\bwhich (?:pokemon|pokémon|move|item|berry|ability|lead)\b",
        r"\b(?:best|safest|better) (?:lead|item|move|option|pokemon|pokémon)\b",
        r"\b(?:likely|recommend|recommendation|counterplay|archetype|meta|trend)\b",
        r"\b(?:biggest threats?|watch out|team strategy|game plan|trying to do)\b",
        r"\bwhat (?:is|are) (?:this|their|the opponent(?:'s)?) team\b",
        r"\bhow (?:do|can|should) (?:i|we)\b",
        r"\b(?:can|will|does) .+\b(?:ko|ohko|2hko|outspeed|survive)\b",
        r"\bhow much damage\b",
        r"\b(?:is|are) .+\b(?:good|better|worth|useful|viable)\b",
        r"\bcommon (?:moves|items|abilities|sets|in tournaments)\b",
    )
    if any(re.search(pattern, text) for pattern in decision_patterns):
        return "strategy"
    if re.match(r"^(?:what (?:is|are|does)|how does|explain|tell me about)\b", text):
        return "battle_info"
    if re.search(r"\b(?:lead|bring|beat|counter|handle|threats?|faster|outspeed|calc|calculate|strateg(?:y|ies))\b", text):
        return "strategy"
    return "battle_info"
