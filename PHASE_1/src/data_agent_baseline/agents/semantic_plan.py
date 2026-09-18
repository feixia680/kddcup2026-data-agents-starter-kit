from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SemanticPlan:
    target_grain: str
    output_fields: list[str]
    filters_and_joins: list[str]
    aggregation: str
    denominator: str
    units: str
    tie_policy: str
    source_scope: str
    checks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        pairs = [
            ("target_grain", self.target_grain),
            ("output_fields", ", ".join(self.output_fields) or "not inferred"),
            ("filters_and_joins", "; ".join(self.filters_and_joins) or "infer from evidence"),
            ("aggregation", self.aggregation),
            ("denominator", self.denominator),
            ("units", self.units),
            ("tie_policy", self.tie_policy),
            ("source_scope", self.source_scope),
            ("checks", "; ".join(self.checks)),
        ]
        return "\n".join(f"- {key}: {value}" for key, value in pairs)


def infer_semantic_plan(question: str) -> SemanticPlan:
    q = question.lower()
    output_fields: list[str] = []
    checks: list[str] = []
    filters: list[str] = []

    if "final score" in q or "score" in q and "final" in q:
        output_fields.extend(["home_team_goals", "away_team_goals"])
        checks.append("split a compact score such as 1-1 into two numeric fields")
    elif "comment" in q or "text" in q:
        output_fields.append("requested comment/text field only")
    elif "post id" in q or "postid" in q:
        output_fields.append("post identifier only")
    elif "transaction" in q or "withdraw" in q:
        output_fields.append("transaction identifier or requested transaction field")

    if any(token in q for token in ("percentage", "percent", "%")):
        output_fields.append("percentage as a numeric value without the percent sign")
        checks.append("keep the denominator explicit and do not emit a percent sign")
    if "average" in q or "mean" in q:
        aggregation = "AVG over the requested population"
        checks.append("use AVG of the raw observations before any requested scaling")
    elif "total" in q or "sum" in q:
        aggregation = "SUM only when the question requests a total"
    elif "count" in q or "how many" in q:
        aggregation = "COUNT the requested entity, not its attributes"
    else:
        aggregation = "row-level selection or direct lookup"

    if "average monthly" in q:
        target_grain = "raw monthly observation before explicit monthly scaling"
    elif "posts" in q and "votes" in q:
        target_grain = "single-user entity counts"
    elif "per unit" in q or "each" in q or "unit price" in q:
        target_grain = "row-level"
        checks.append("compute the per-row ratio before filtering; do not filter the raw total")
    elif any(token in q for token in ("lowest", "highest", "cheapest", "most expensive", "closest")):
        target_grain = "row-level argmin/argmax"
        checks.append("compare individual rows before grouping")
    elif any(token in q for token in ("by country", "by type", "for each", "grouped")):
        target_grain = "group-level"
    else:
        target_grain = "scalar or requested entity"

    if "average monthly" in q:
        checks.append("compute AVG over the filtered raw monthly records, then apply the explicit /12 scaling exactly once")
        checks.append("do not replace AVG(raw records) with SUM(raw records) / 12")
    if "posts" in q and "votes" in q:
        output_fields.append("post_count divided by vote_count for the same user")
        checks.append("count the user's posts and votes as separate entities before division")
        checks.append("do not count votes attached to posts as the user's received-vote total")
    if "rank" in q:
        checks.append("use the semantic rank field; do not substitute a display position")
        filters.append("rank is distinct from position")
    if "position" in q:
        checks.append("use position only when the question explicitly asks for position")
    if "1k" in q or "sample" in q or "subset" in q:
        source_scope = "treat sample files as evidence only; verify whether a full source exists"
        checks.append("do not infer that a *_1k or sample file is the complete dataset")
    else:
        source_scope = "all available records matching the task"

    denominator = "the population explicitly named by the question"
    if "average monthly" in q:
        aggregation = "AVG(filtered raw monthly consumption) / 12"
        denominator = "the explicit monthly scaling factor 12, applied after AVG"
    elif "posts" in q and "votes" in q:
        aggregation = "COUNT(user posts) / COUNT(user votes)"
        denominator = "the user's vote count, not the number of posts or post-linked votes"
    elif "per" in q and "total" in q:
        denominator = "the per-row amount/quantity, not the aggregate total"
    elif "percentage" in q or "percent" in q:
        denominator = "the explicitly requested numerator and denominator"

    units = "preserve source units and numeric precision"
    if "second" in q or "time" in q:
        units = "preserve the requested time unit; convert only once"

    tie_policy = "return all ties unless the question asks for one deterministic row"
    if "the" in q and any(token in q for token in ("lowest", "highest", "closest")):
        tie_policy = "return all tied rows, then apply an explicit deterministic tie-break only if needed"

    return SemanticPlan(
        target_grain=target_grain,
        output_fields=list(dict.fromkeys(output_fields)),
        filters_and_joins=filters,
        aggregation=aggregation,
        denominator=denominator,
        units=units,
        tie_policy=tie_policy,
        source_scope=source_scope,
        checks=list(dict.fromkeys(checks)),
    )
