from typing import List, Dict, Counter
from abc import ABC, abstractmethod
from feed_reader import Article
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class ArticleRanker(ABC):
    @abstractmethod
    def rank_articles(self, articles: List[Article]) -> List[Article]:
        """Rank articles according to some criteria."""
        pass

class TitleEmbeddingDiversityRanker(ArticleRanker):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the diversity ranker with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def rank_articles(self, articles: List[Article]) -> List[Article]:
        """
        Rank articles by diversity using an iterative approach.
        At each step, we identify the least unique article (highest mean similarity)
        and move it to the end of the output list.
        """
        if not articles:
            return []

        # Compute embeddings for all articles
        titles = [article.title for article in articles]
        embeddings = self.model.encode(titles, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()

        # Initialize lists for tracking
        remaining_indices = list(range(len(articles)))
        ranked_indices = []
        
        # Compute initial similarity matrix
        similarity_matrix = np.zeros((len(articles), len(articles)))
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                similarity = self._compute_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        # Iteratively select least unique articles
        while remaining_indices:
            # Compute mean similarity for each remaining article
            mean_similarities = []
            for idx in remaining_indices:
                # Get similarities to all other remaining articles
                similarities = [similarity_matrix[idx, j] for j in remaining_indices if j != idx]
                mean_similarities.append(np.mean(similarities))

            # Find the least unique article (highest mean similarity)
            least_unique_idx = remaining_indices[np.argmax(mean_similarities)]
            
            # Move it to the end of ranked list
            ranked_indices.append(least_unique_idx)
            remaining_indices.remove(least_unique_idx)

        # Return articles in ranked order
        return [articles[i] for i in ranked_indices]

class SourceDiversityRanker(ArticleRanker):
    def __init__(self, max_articles: int):
        """Initialize the source diversity ranker with the maximum number of articles to consider."""
        self.max_articles = max_articles
        self.max_source_count = max(max_articles // 5, 3)  # m = max(N/5, 3)

    def _get_source_counts(self, articles: List[Article]) -> Counter:
        """Get the count of articles from each source."""
        return Counter(article.source for article in articles)

    def _find_least_represented_source(self, articles: List[Article], top_n: List[Article]) -> str:
        """
        Find the source with the fewest articles in the top N.
        If there are sources with 0 in top N but >1 overall, pick the highest ranked of those.
        """
        top_n_sources = Counter(article.source for article in top_n)
        all_sources = self._get_source_counts(articles)
        
        # First try to find sources with 0 in top N but >1 overall
        zero_in_top = {source: count for source, count in all_sources.items() 
                      if source not in top_n_sources and count > 1}
        
        if zero_in_top:
            # Return the source with the highest overall count
            return max(zero_in_top.items(), key=lambda x: x[1])[0]
        
        # Otherwise return the source with the fewest in top N
        return min(top_n_sources.items(), key=lambda x: x[1])[0]

    def _find_highest_ranked_from_source(self, articles: List[Article], source: str) -> int:
        """Find the index of the highest ranked article from the given source."""
        for i, article in enumerate(articles):
            if article.source == source:
                return i
        return -1

    def rank_articles(self, articles: List[Article]) -> List[Article]:
        """
        Rank articles to ensure source diversity in the top N articles.
        """
        if not articles:
            return []

        # Start with the original order
        ranked_articles = articles.copy()
        iterations = 0

        while iterations < self.max_articles:
            # Get the top N articles
            top_n = ranked_articles[:self.max_articles]
            
            # Count sources in top N
            source_counts = Counter(article.source for article in top_n)
            
            # Check if any source has too many articles
            overrepresented_sources = {source: count for source, count in source_counts.items() 
                                    if count > self.max_source_count}
            
            if not overrepresented_sources:
                break
                
            # Find the first overrepresented source
            source_to_move = next(iter(overrepresented_sources))
            
            # Find the first article from this source in top N
            idx_to_move = next(i for i, article in enumerate(top_n) 
                             if article.source == source_to_move)
            
            # Move it to the end
            article_to_move = ranked_articles.pop(idx_to_move)
            ranked_articles.append(article_to_move)
            
            # Find the least represented source
            least_represented = self._find_least_represented_source(ranked_articles, top_n)
            
            # Find the highest ranked article from the least represented source
            idx_to_promote = self._find_highest_ranked_from_source(ranked_articles, least_represented)
            
            if idx_to_promote >= self.max_articles:
                # Move it to the front
                article_to_promote = ranked_articles.pop(idx_to_promote)
                ranked_articles.insert(0, article_to_promote)
            
            iterations += 1
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Iteration {iterations}:")
                logger.debug(f"Source counts in top {self.max_articles}: {dict(source_counts)}")
                logger.debug(f"Moved article from {source_to_move} to end")
                if idx_to_promote >= self.max_articles:
                    logger.debug(f"Promoted article from {least_represented} to front")

        return ranked_articles 