import os
from dotenv import load_dotenv
from feed_reader import FeedReader, RSSSource
from article_processor import (
    ArticleProcessor, KeywordFilter, MaxArticlesFilter, RedundancyFilter,
    ArticleRanker as ArticleRankerProcessor, NegativeKeywordFilter, NoveltyFilter
)
from llm_processor import LLMProcessor, ProcessedArticle
from markdown_generator import MarkdownGenerator
from content_fetcher import ContentFetcher
from article_ranker import TitleEmbeddingDiversityRanker, SourceDiversityRanker
import logging
import json
from datetime import datetime
from typing import List, Optional
from feed_reader import Article

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("config.json not found")
        return None

def process_articles_sequentially(
    articles: List[Article],
    content_fetcher: ContentFetcher,
    llm_processor: LLMProcessor,
    dry_run: bool = False
) -> List[ProcessedArticle]:
    """Process articles one at a time, fetching content and generating insights."""
    processed_articles = []
    
    for i, article in enumerate(articles, 1):
        logger.info(f"\nProcessing article {i}/{len(articles)}: {article.title}")
        
        if dry_run:
            logger.info("Dry run mode - skipping content fetch and LLM processing")
            # Create a dummy processed article for dry run
            processed = ProcessedArticle(
                article,
                "Summary not generated in dry run mode",
                "Insights not generated in dry run mode"
            )
        else:
            # Fetch content for main article (including comments if available)
            logger.info(f"Fetching content from {article.link}")
            if article.comments_link:
                logger.info(f"Comments link found: {article.comments_link}")
            main_content, main_image = content_fetcher.fetch_article_with_comments(article)

            if main_image:
                article.image_url = main_image

            # Fetch content for related articles
            related_contents = []
            for related in article.related_links:
                logger.info(f"Fetching related article: {related.title}")
                if content := content_fetcher.fetch_article_with_comments(related):
                    related_content, related_image = content
                    related_contents.append(f"Related Article from {related.link}:\n{related_content}")
                    if related_image and not article.image_url:  # Use first available image
                        article.image_url = related_image
            
            # Combine all content
            full_content = f"Main Article from {article.link}:\n{main_content}\n\n"
            if related_contents:
                full_content += "\n".join(related_contents)
            
            # Process with LLM
            logger.info("Generating insights...")
            processed = llm_processor.process_article(article, full_content)
        
        processed_articles.append(processed)
        logger.info(f"✓ Completed processing: {article.title}")
    
    return processed_articles

def main():
    # Load configuration
    config = load_config()
    if not config:
        return

    # Initialize components
    feed_reader = FeedReader()
    article_processor = ArticleProcessor(verbose=config.get('verbose', False))
    llm_processor = LLMProcessor()
    markdown_generator = MarkdownGenerator()
    content_fetcher = ContentFetcher(verbose=config.get('verbose', False))

    # Add RSS sources
    for source in config.get('rss_sources', []):
        feed_reader.add_source(RSSSource(source['url'], source['name']))

    # Add negative keyword filter (before positive keywords)
    article_processor.add_processor(NegativeKeywordFilter(config.get('negative_keywords', [])))

    # Add keyword filter
    article_processor.add_processor(KeywordFilter(config.get('keywords', [])))
    
    # Add novelty filter
    article_processor.add_processor(NoveltyFilter(
        output_dir=config.get('output_dir', 'output'),
        lookback_days=config.get('novelty_lookback_days', 7),
        similarity_threshold=config.get('novelty_similarity_threshold', 0.85)
    ))
    
    # Add redundancy filter
    article_processor.add_processor(RedundancyFilter(
        similarity_threshold=config.get('similarity_threshold', 0.85)
    ))
    
    # Add rankers
    article_processor.add_processor(ArticleRankerProcessor(TitleEmbeddingDiversityRanker()))
    max_articles = config.get('max_articles', 5)  # Default to 5 if not specified
    article_processor.add_processor(ArticleRankerProcessor(SourceDiversityRanker(max_articles)))
    
    # Add max articles filter (after ranking)
    article_processor.add_processor(MaxArticlesFilter(max_articles))

    # Fetch and process articles
    logger.info("Fetching articles...")
    articles = feed_reader.fetch_all_articles()
    
    logger.info("Processing articles...")
    processed_articles = article_processor.process_articles(articles)
    
    # Check if we're in dry run mode
    dry_run = config.get('dry_run', False)
    if dry_run:
        logger.info("Running in DRY RUN mode - no content fetching or LLM processing will occur")
    
    logger.info("Processing articles sequentially...")
    processed_articles = process_articles_sequentially(
        processed_articles,
        content_fetcher,
        llm_processor,
        dry_run
    )
    
    if not dry_run:
        # Generate markdown
        logger.info("Generating markdown...")
        markdown_content = markdown_generator.generate_markdown(processed_articles)
        
        # Save to file
        output_dir = config.get('output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"reading_list_{datetime.now().strftime('%Y%m%d')}.md")
        markdown_generator.save_markdown(markdown_content, output_file)
    else:
        logger.info("Dry run complete - no markdown generated")

if __name__ == "__main__":
    main() 