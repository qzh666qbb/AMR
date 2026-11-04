from __future__ import annotations

import re
from typing import Set


class MockScorer:
    def __init__(self, stop_words: Set[str] = None):
        """
        模拟 AMR 解析和三元组匹配的极速打分器。
        使用简单的概念词袋（Concept Bag-of-Words）计算 Jaccard 相似度。
        """
        if stop_words is None:
            self.stop_words = {
                "a", "an", "and", "at", "he", "in", "is", "it", "of",
                "on", "or", "she", "the", "they", "to", "we",
            }
        else:
            self.stop_words = stop_words

    def extract_concepts(self, text: str) -> Set[str]:
        """
        模拟 AMR 骨架提取，将文本转化为核心概念集合。
        """
        # 转小写并去除非字母字符
        words = re.findall(r'\b[a-z]+\b', text.lower())
        # 过滤停用词
        concepts = {w for w in words if w not in self.stop_words}
        return concepts

    def score(self, span_text: str, target_text: str) -> float:
        """
        计算两个文本片段的概念词袋重合度 (Jaccard 相似度)。
        """
        span_concepts = self.extract_concepts(span_text)
        target_concepts = self.extract_concepts(target_text)
        
        if not span_concepts and not target_concepts:
            return 0.0
            
        intersection = span_concepts.intersection(target_concepts)
        union = span_concepts.union(target_concepts)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
