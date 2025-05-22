from typing import List, Dict, Any, Optional
import feedparser
from abc import ABC, abstractmethod
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Article:
    def __init__(self, title: str, link: str, published: datetime, summary: str, source: str):
        self.title = title
        self.link = link
        self.published = published
        self.summary = summary
        self.source = source
        self.related_links: List['Article'] = []
        self.image_url: Optional[str] = None
        self.comments_link: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "link": self.link,
            "published": self.published.isoformat(),
            "summary": self.summary,
            "source": self.source,
            "related_links": [rel.to_dict() for rel in self.related_links],
            "image_url": self.image_url,
            "comments_link": self.comments_link
        }

    @classmethod
    def from_feed_entry(cls, entry: Dict[str, Any], source: str) -> 'Article':
        comments_link = None
        if hasattr(entry, 'comments'):
            comments_link = entry.comments
        elif hasattr(entry, 'comments_link'):
            comments_link = entry.comments_link
        elif hasattr(entry, 'comments_url'):
            comments_link = entry.comments_url

        article = cls(
            title=entry.title,
            link=entry.link,
            published=entry.published_parsed,
            summary=entry.summary,
            source=source
        )
        article.comments_link = comments_link
        return article

class ContentSource(ABC):
    @abstractmethod
    def fetch_articles(self) -> List[Article]:
        pass

class RSSSource(ContentSource):
    def __init__(self, url: str, source_name: str):
        self.url = url
        self.source_name = source_name

    def fetch_articles(self) -> List[Article]:
        try:
            feed = feedparser.parse(self.url)
            articles = []
            
            for entry in feed.entries:
                article = Article.from_feed_entry(entry, self.source_name)
                articles.append(article)
            
            logger.info(f"Successfully fetched {len(articles)} articles from {self.source_name}")
            return articles
        except Exception as e:
            logger.error(f"Error fetching from {self.source_name}: {str(e)}")
            return []

class FeedReader:
    def __init__(self):
        self.sources: List[ContentSource] = []

    def add_source(self, source: ContentSource):
        self.sources.append(source)

    def fetch_all_articles(self) -> List[Article]:
        all_articles = []
        for source in self.sources:
            articles = source.fetch_articles()
            all_articles.extend(articles)
        return all_articles 