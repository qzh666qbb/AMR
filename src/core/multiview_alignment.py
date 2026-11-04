"""对应选择只用文本/事件线索；结构Bank分数不进入优化器。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .multiview_views import Event


@dataclass(frozen=True)
class AlignmentConfig:
    min_similarity: float = 0.35
    max_events_per_candidate: int = 3
    semantic_weight: float = 0.8
    position_weight: float = 0.2
    solver_time_limit: float = 15.0

    def __post_init__(self):
        if (not 0 <= self.min_similarity <= 1 or self.max_events_per_candidate < 1
                or min(self.semantic_weight, self.position_weight) < 0
                or self.solver_time_limit <= 0):
            raise ValueError("对应参数无效")


def overlap_fraction(left: dict, right: dict) -> float:
    overlap = max(0, min(left["end_char"], right["end_char"])
                  - max(left["start_char"], right["start_char"]))
    minimum = min(left["end_char"] - left["start_char"],
                  right["end_char"] - right["start_char"])
    return overlap / minimum if minimum else 0.0


def _role_name(value):
    match = re.search(r"\[([^\]]+)\]", value)
    return match.group(1).lower() if match else value.split(" :", 1)[0].lower()


def event_conflicts(left: Event, right: Event) -> list:
    if left.predicate != right.predicate:
        return []
    reasons = []
    if left.negated != right.negated:
        reasons.append("negation_conflict")
    if left.numbers and right.numbers and set(left.numbers) != set(right.numbers):
        reasons.append("number_conflict")
    lroles, rroles = dict(left.roles), dict(right.roles)
    for role in (":ARG0", ":ARG1", ":ARG2"):
        if role in lroles and role in rroles and _role_name(lroles[role]) != _role_name(rroles[role]):
            reasons.append("role_conflict:" + role)
    return reasons


def surface_conflicts(surface: dict, event: Event) -> list:
    """只在同谓词有明确矛盾时拒配；缺线索保持未知。"""
    predicate = re.sub(r"-\d\d$", "", event.predicate)
    clauses = [clause for clause in surface["clauses"] if clause["predicate"] == predicate]
    if not clauses:
        return []
    alternatives = []
    for clause in clauses:
        reasons = []
        if clause["negated"] != event.negated:
            reasons.append("surface_negation_conflict")
        if (clause.get("numbers") and event.numbers
                and set(clause["numbers"]) != set(event.numbers)):
            reasons.append("surface_number_conflict")
        for role, value in event.roles:
            if role not in clause["roles"] or role not in (":ARG0", ":ARG1"):
                continue
            name = _role_name(value)
            observed = clause["roles"][role]
            # 仅显式名字能作硬拒配；普通concept/代词不进行字符串强等。
            if "[" in value and name not in observed:
                reasons.append("surface_role_conflict:" + role)
        alternatives.append(reasons)
    return [] if any(not reasons for reasons in alternatives) else sorted(set(
        reason for reasons in alternatives for reason in reasons))


def core_identity(event: Event) -> tuple:
    """保留全部核心角色、否定、实体和数值；仅忽略非核心修饰差异。"""
    return (event.predicate, event.negated,
            tuple((role, value) for role, value in event.roles if re.fullmatch(r":ARG\d+", role)),
            event.entities, event.numbers)


def event_groups(events: list, candidates=None) -> list:
    """核心关系相同且支持区间重叠才容许修饰差异；全组逐对核验。"""
    groups = []
    for event in events:
        if event.missing_core_roles:
            continue
        def compatible(group):
            for member in group["events"]:
                if member.signature == event.signature:
                    continue
                if candidates is None or core_identity(member) != core_identity(event):
                    return False
                if overlap_fraction(candidates[member.candidate_index],
                                    candidates[event.candidate_index]) < 0.5:
                    return False
                # 时间、条件等非核心关系不允许因简化身份而抹去。
                protected = (":time", ":condition", ":concession", ":location", ":polarity")
                if {role: value for role, value in member.roles if role in protected} != {
                        role: value for role, value in event.roles if role in protected}:
                    return False
            return True
        match = next((group for group in groups if compatible(group)), None)
        if match is None:
            groups.append({"group_id": f"g{len(groups)}", "signature": event.signature,
                           "events": [event]})
        else:
            match["events"].append(event)
    return groups


def align(candidates: list, events: list, text_vectors: np.ndarray,
          event_vectors: np.ndarray, surfaces: list,
          config: AlignmentConfig = AlignmentConfig(), mode="predicted") -> dict:
    if mode not in ("predicted", "position", "semantic"):
        raise ValueError("未知对应模式")
    if len(text_vectors) != len(candidates) or len(event_vectors) != len(events):
        raise ValueError("视图数量不匹配")
    groups = event_groups(events, candidates)
    event_index = {event.event_id: index for index, event in enumerate(events)}
    edges, rejected = [], []
    for ci, candidate in enumerate(candidates):
        for gi, group in enumerate(groups):
            options = []
            for event in group["events"]:
                support = candidates[event.candidate_index]
                overlap = overlap_fraction(candidate, support)
                if overlap <= 0 and candidate["span_sha256"] != support["span_sha256"]:
                    continue
                similarity = float(text_vectors[ci] @ event_vectors[event_index[event.event_id]])
                reasons = surface_conflicts(surfaces[ci], event) if mode == "predicted" else []
                if reasons:
                    rejected.append({"candidate_index": ci, "group_id": group["group_id"],
                                     "event_id": event.event_id, "reasons": reasons})
                    continue
                if mode != "position" and similarity < config.min_similarity:
                    continue
                # position模式直接使用事件原片段，作为已知位置的简单对照。
                if mode == "position" and ci != event.candidate_index:
                    continue
                objective = (overlap if mode == "position" else similarity if mode == "semantic"
                             else config.semantic_weight * similarity + config.position_weight * overlap)
                options.append({
                    "candidate_index": ci, "group_index": gi, "group_id": group["group_id"],
                    "event_id": event.event_id, "event_candidate_index": event.candidate_index,
                    "event_support_word_count": support.get("word_count", len(support.get("text", "").split())),
                    "similarity": similarity, "overlap": overlap, "objective": objective,
                    "cross_candidate": ci != event.candidate_index,
                })
            if options:
                # 每个文本/事件组只保留一个最强观测；不能重复占多个容量。
                # 相同对应线索下优先完整的短支持图，避免任意beam ID把好片段
                # 替换成带其他事件的长图；排序不读取Bank分数。
                options.sort(key=lambda edge: (-edge["objective"], edge["event_support_word_count"],
                                               edge["event_candidate_index"], edge["event_id"]))
                edges.append(options[0])
    selected = []
    status = "no_feasible_correspondence"
    if edges:
        incompatible = []
        if mode == "predicted":
            for left_index, left in enumerate(edges):
                for right_index in range(left_index + 1, len(edges)):
                    right = edges[right_index]
                    if (left["candidate_index"] == right["candidate_index"]
                            and event_conflicts(groups[left["group_index"]]["events"][0],
                                                groups[right["group_index"]]["events"][0])):
                        incompatible.append((left_index, right_index))
        base_rows = len(candidates) + len(groups)
        constraints = lil_matrix((base_rows + len(incompatible), len(edges)))
        for index, edge in enumerate(edges):
            constraints[edge["candidate_index"], index] = 1
            constraints[len(candidates) + edge["group_index"], index] = 1
        for row, (left, right) in enumerate(incompatible, start=base_rows):
            constraints[row, left] = constraints[row, right] = 1
        upper = np.array([config.max_events_per_candidate] * len(candidates)
                         + [1] * (len(groups) + len(incompatible)))
        # 微小索引项只用于同目标的稳定选择，不添加Bank信息。
        objective = np.array([-edge["objective"] + index * 1e-10
                              for index, edge in enumerate(edges)])
        result = milp(
            objective, integrality=np.ones(len(edges)), bounds=Bounds(0, 1),
            constraints=LinearConstraint(constraints.tocsr(), 0, upper),
            options={"time_limit": config.solver_time_limit, "mip_rel_gap": 0.0})
        if not result.success:
            raise RuntimeError(f"对应优化未求得最优解，status={result.status}: {result.message}")
        selected = [dict(edge) for edge, flag in zip(edges, result.x) if flag > 0.5]
        status = "optimal"
    for edge in selected:
        alternatives = [other["objective"] for other in edges
                        if other["candidate_index"] == edge["candidate_index"]
                        and other["group_index"] != edge["group_index"]]
        edge["semantic_margin"] = edge["objective"] - max(alternatives, default=0.0)
        # 规则支持度不解释为正确对应概率；不使用Bank分数。
        edge["rule_weight"] = float(np.clip(
            max(0.0, edge["similarity"]) * (0.5 + 0.5 * edge["overlap"]), 0, 1))
    return {
        "mode": mode, "config": asdict(config), "solver_status": status,
        "objective": sum(edge["objective"] for edge in selected),
        "groups": [{"group_id": group["group_id"], "signature": group["signature"],
                    "event_ids": [event.event_id for event in group["events"]]}
                   for group in groups],
        "edges": edges, "selected": selected, "rejected": rejected,
        "unmatched_groups": [group["group_id"] for group in groups
                             if not any(edge["group_id"] == group["group_id"] for edge in selected)],
        "unmatched_candidates": [index for index in range(len(candidates))
                                if not any(edge["candidate_index"] == index for edge in selected)],
    }
