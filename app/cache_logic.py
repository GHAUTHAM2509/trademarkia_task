import numpy as np
from typing import Dict, List, Tuple

class ClusterAwareSemanticCache:
    def __init__(self, similarity_threshold: float = 0.86):
        self.similarity_threshold = similarity_threshold
        
        # Format: { cluster_id: [(query_vector, original_query_string, results_list), ...] }
        self.cache: Dict[int, List[Tuple[np.ndarray, str, list]]] = {}
        
        # Metrics
        self.hits = 0
        self.misses = 0

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates cosine similarity between two normalized vectors."""
        # For normalized vectors, dot product equals cosine similarity
        return float(np.dot(v1, v2))

    def check_cache(self, query_vector: np.ndarray, cluster_id: int) -> Tuple[list, float, str]:
        """
        Checks if a mathematically identical or paraphrased query 
        already exists in the relevant cluster bucket.
        Returns: (best_results, similarity_score, matched_query_string)
        """
        if cluster_id not in self.cache:
            self.misses += 1
            return None, 0.0, None
            
        bucket = self.cache[cluster_id]
        best_similarity = -1.0
        best_results = None
        best_query_string = None
        
        for cached_vector, cached_query_string, cached_results in bucket:
            sim = self._cosine_similarity(query_vector, cached_vector)
            if sim > best_similarity:
                best_similarity = sim
                best_results = cached_results
                best_query_string = cached_query_string
                
        if best_similarity >= self.similarity_threshold:
            print(f"⚡ CACHE HIT! (Similarity: {best_similarity:.3f})")
            self.hits += 1
            return best_results, best_similarity, best_query_string
            
        print(f"CACHE MISS (Best similarity: {best_similarity:.3f})")
        self.misses += 1
        return None, best_similarity, None

    def add_to_cache(self, query_vector: np.ndarray, original_query_string: str, cluster_id: int, results: list):
        """
        Adds a completely new query and its database results to the cache.
        """
        if cluster_id not in self.cache:
            self.cache[cluster_id] = []
        
        self.cache[cluster_id].append((query_vector, original_query_string, results))

    def get_stats(self) -> dict:
        total_entries = sum(len(bucket) for bucket in self.cache.values())
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0
        return {
            "total_entries": total_entries,
            "hit_count": self.hits,
            "miss_count": self.misses,
            "hit_rate": round(hit_rate, 3)
        }
        
    def clear_cache(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


"""
Explaination:
    This module implements a similarity-based caching mechanism designed to optimize query performance by reusing results from semantically similar previous queries. 
    Key features include:
    Vector-based similarity matching with a configurable threshold.
    Bucketed storage using cluster IDs to narrow search space.
    Performance tracking (hits, misses, and hit rate).
    Methods for cache population, retrieval, and maintenance.

"""