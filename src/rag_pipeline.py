from __future__ import annotations

from typing import Dict, List


SYSTEM_RULES = """Sen Türk hukuk alanında kaynak destekli cevap veren bir yardımcı sistemsin.
Sadece verilen bağlamdaki bilgilere dayan.
Bağlamda cevap yoksa bunu açıkça söyle.
Her cevapta ilgili kaynak ve madde numarasını belirt.
Hukuki danışmanlık verdiğini iddia etme; metinleri araştırma amaçlı açıkla.
"""


def format_contexts(hits: List[Dict], max_chars_per_context: int = 1800) -> str:
    blocks = []
    for hit in hits:
        context = str(hit.get("context", ""))[:max_chars_per_context]
        blocks.append(
            f"[Kaynak: {hit.get('kaynak')} | Madde: {hit.get('madde_no')} | Rank: {hit.get('rank')} ]\n{context}"
        )
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, hits: List[Dict]) -> str:
    contexts = format_contexts(hits)
    return f"""{SYSTEM_RULES}

BAĞLAM:
{contexts}

SORU:
{question}

CEVAP:
"""
