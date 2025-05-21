from typing import List
from abc import ABC, abstractmethod
from feed_reader import Article
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ArticleRanker(ABC):
    @abstractmethod
    def rank_articles(self, articles: List[Article]) -> List[Article]:
        """Rank articles according to some criteria."""
        pass

class DiversityRanker(ArticleRanker):
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