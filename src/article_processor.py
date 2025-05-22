from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from feed_reader import Article
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
from article_ranker import ArticleRanker

logger = logging.getLogger(__name__)

class ArticleProcessor(ABC):
    @abstractmethod
    def process_articles(self, articles: List[Article], verbose: bool = False) -> List[Article]:
        pass

class ArticleFilter(ArticleProcessor):
    @abstractmethod
    def filter_articles(self, articles: List[Article], verbose: bool = False) -> List[Article]:
        pass

    def process_articles(self, articles: List[Article], verbose: bool = False) -> List[Article]:
        return self.filter_articles(articles, verbose)

class ArticleRanker(ArticleProcessor):
    def __init__(self, ranker: ArticleRanker):
        self.ranker = ranker

    def process_articles(self, articles: List[Article], verbose: bool = False) -> List[Article]:
        if verbose:
            logger.info(f"\nApplying {self.ranker.__class__.__name__}...")
        return self.ranker.rank_articles(articles)

class KeywordFilter(ArticleFilter):
    def __init__(self, keywords: List[str]):
        self.keywords = [k.lower() for k in keywords]

    def filter_articles(self, articles: List[Article], verbose: bool = False) -> List[Article]:
        filtered_articles = []
        for article in articles:
            # Check if any keyword is in the title or summary
            matches = [keyword for keyword in self.keywords 
                      if keyword in article.title.lower() or keyword in article.summary.lower()]
            
            if matches:
                filtered_articles.append(article)
                if verbose:
                    logger.info(f"✓ Article matched: '{article.title}'")
                    logger.info(f"  Matched keywords: {', '.join(matches)}")
            elif verbose:
                logger.info(f"✗ Article skipped: '{article.title}'")
                logger.info(f"  No keyword matches found")
        
        logger.info(f"Filtered {len(articles)} articles down to {len(filtered_articles)} using keywords")
        return filtered_articles

class RedundancyFilter(ArticleFilter):
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        # Use a small, fast model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def filter_articles(self, articles: List[Article], verbose: bool = False) -> List[Article]:
        if not articles:
            return []

        # Compute embeddings for all articles once
        titles = [article.title for article in articles]
        embeddings = self.model.encode(titles, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()  # Convert to numpy for easier computation

        # Keep track of which articles to keep
        to_keep = []
        processed_indices = set()

        if verbose:
            logger.info("\nComputing BERTScore similarities between articles:")
            logger.info("=" * 80)

        for i, (article, emb1) in enumerate(zip(articles, embeddings)):
            if i in processed_indices:
                continue

            to_keep.append(article)
            processed_indices.add(i)

            if verbose:
                logger.info(f"\nComparing with primary article: '{article.title}'")

            # Check similarity with remaining articles
            for j, (other, emb2) in enumerate(zip(articles[i+1:], embeddings[i+1:]), i+1):
                if j in processed_indices:
                    continue

                similarity = self._compute_similarity(emb1, emb2)
                
                if verbose:
                    status = "✓" if similarity >= self.similarity_threshold else "✗"
                    logger.info(f"{status} BERTScore: {similarity:.3f} - '{other.title}'")
                
                if similarity >= self.similarity_threshold:
                    processed_indices.add(j)
                    article.related_links.append(other)
                    if verbose:
                        logger.info(f"  → Added as related article")

        logger.info(f"\nRedundancy filter: {len(articles)} articles -> {len(to_keep)} unique articles")
        return to_keep

class MaxArticlesFilter(ArticleFilter):
    def __init__(self, max_articles: int):
        self.max_articles = max_articles

    def filter_articles(self, articles: List[Article], verbose: bool = False) -> List[Article]:
        # Take the first N articles
        limited_articles = articles[:self.max_articles]
        
        if verbose:
            logger.info(f"\nApplying max articles limit ({self.max_articles}):")
            for i, article in enumerate(articles):
                status = "✓" if i < self.max_articles else "✗"
                logger.info(f"{status} Article {i+1}: '{article.title}'")
        
        logger.info(f"Limited articles from {len(articles)} to {len(limited_articles)} (max: {self.max_articles})")
        return limited_articles

class NegativeKeywordFilter(ArticleFilter):
    def __init__(self, negative_keywords: List[str]):
        self.negative_keywords = [k.lower() for k in negative_keywords]

    def filter_articles(self, articles: List[Article], verbose: bool = False) -> List[Article]:
        filtered_articles = []
        for article in articles:
            # Check if any negative keyword is in the title or summary
            matches = [keyword for keyword in self.negative_keywords 
                      if keyword in article.title.lower() or keyword in article.summary.lower()]
            
            if not matches:  # Keep articles that don't contain negative keywords
                filtered_articles.append(article)
                if verbose:
                    logger.info(f"✓ Article kept: '{article.title}'")
            elif verbose:
                logger.info(f"✗ Article removed: '{article.title}'")
                logger.info(f"  Matched negative keywords: {', '.join(matches)}")
        
        logger.info(f"Negative keyword filter: {len(articles)} articles -> {len(filtered_articles)} articles")
        return filtered_articles

class ArticleProcessor:
    def __init__(self, verbose: bool = False):
        self.processors: List[ArticleProcessor] = []
        self.verbose = verbose

    def add_processor(self, processor: ArticleProcessor):
        self.processors.append(processor)

    def process_articles(self, articles: List[Article]) -> List[Article]:
        processed_articles = articles
        for processor in self.processors:
            if self.verbose:
                logger.info("\n" + "="*50)
            processed_articles = processor.process_articles(processed_articles, self.verbose)
        return processed_articles 