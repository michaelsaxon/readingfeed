from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from feed_reader import Article
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
from article_ranker import ArticleRanker
import os
import re
from datetime import datetime, timedelta

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

class NoveltyFilter(ArticleFilter):
    def __init__(self, output_dir: str, lookback_days: int = 7, similarity_threshold: float = 0.85):
        self.output_dir = output_dir
        self.lookback_days = lookback_days
        self.similarity_threshold = similarity_threshold
        # Use the same model as RedundancyFilter for consistency
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def _get_previous_headlines(self) -> List[str]:
        """Get headlines from previous reports within lookback period."""
        headlines = []
        today = datetime.now()
        
        # Scan previous reports
        for i in range(self.lookback_days):
            date = today - timedelta(days=i)
            filename = f"reading_list_{date.strftime('%Y%m%d')}.md"
            filepath = os.path.join(self.output_dir, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Extract h2 level headers (article titles)
                    matches = re.finditer(r'^## (.+)$', content, re.MULTILINE)
                    headlines.extend(match.group(1) for match in matches)
        
        return headlines

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def filter_articles(self, articles: List[Article], verbose: bool = False) -> List[Article]:
        if not articles:
            return []

        # Get previous headlines
        previous_headlines = self._get_previous_headlines()
        if not previous_headlines:
            if verbose:
                logger.info("No previous headlines found for novelty comparison")
            return articles

        # Compute embeddings for previous headlines
        prev_embeddings = self.model.encode(previous_headlines, convert_to_tensor=True)
        prev_embeddings = prev_embeddings.cpu().numpy()

        # Filter articles
        filtered_articles = []
        for article in articles:
            # Compute embedding for current article title
            title_embedding = self.model.encode([article.title], convert_to_tensor=True)
            title_embedding = title_embedding.cpu().numpy()[0]

            # Check similarity with all previous headlines
            max_similarity = max(
                self._compute_similarity(title_embedding, prev_emb)
                for prev_emb in prev_embeddings
            )

            if max_similarity < self.similarity_threshold:
                filtered_articles.append(article)
                if verbose:
                    logger.info(f"✓ Novel article: '{article.title}'")
                    logger.info(f"  Max similarity with previous: {max_similarity:.3f}")
            elif verbose:
                logger.info(f"✗ Similar to previous: '{article.title}'")
                logger.info(f"  Max similarity: {max_similarity:.3f}")

        logger.info(f"Novelty filter: {len(articles)} articles -> {len(filtered_articles)} novel articles")
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