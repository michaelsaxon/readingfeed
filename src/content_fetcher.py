import requests
from bs4 import BeautifulSoup
import logging
import time
from typing import Optional, Tuple
from urllib.parse import urlparse, urlunparse
import random
import os
import hashlib

logger = logging.getLogger(__name__)

class ContentFetcher:
    def __init__(self, delay_between_requests: float = 1.0, verbose: bool = False):
        self.delay_between_requests = delay_between_requests
        self.last_request_time = 0
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Setup cache directory if in verbose mode
        if self.verbose:
            self.cache_dir = "cache"
            if os.path.exists(self.cache_dir):
                # Clear existing cache
                for file in os.listdir(self.cache_dir):
                    os.remove(os.path.join(self.cache_dir, file))
            else:
                os.makedirs(self.cache_dir)

    def _get_cache_path(self, url: str) -> str:
        """Generate cache file path for a URL."""
        # Create a hash of the URL for the filename
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.txt")

    def _save_to_cache(self, url: str, content: str):
        """Save content to cache file."""
        if not self.verbose:
            return
            
        cache_path = self._get_cache_path(url + content[:min(100, len(content))])
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Cached content to {cache_path} (length: {len(content)} chars)")
        except Exception as e:
            logger.error(f"Failed to cache content for {url}: {str(e)}")

    def _clean_url(self, url: str) -> str:
        """Clean URL by removing unnecessary query parameters and fragments."""
        # Parse the URL
        parsed = urlparse(url)
        
        # Remove query parameters and fragments
        cleaned = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            '',  # params
            '',  # query
            ''   # fragment
        ))
        
        return cleaned

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc

    def _extract_content(self, soup: BeautifulSoup, domain: str) -> Optional[str]:
        """Extract main content based on domain-specific rules."""
        # Common content container classes/ids
        content_selectors = [
            'article', 'main', '.post-content', '.article-content',
            '.story-body', '.entry-content', '#content', '.content'
        ]
        
        # Try domain-specific rules first
        if 'nytimes.com' in domain:
            content = soup.select_one('article')
        elif 'techcrunch.com' in domain:
            content = soup.select_one('main.template-content')
        elif 'vox.com' in domain:
            content = soup.select_one('.c-entry-content')
        elif 'theatlantic.com' in domain:
            content = soup.select_one('article')
        else:
            # Generic fallback: try common selectors
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break

        if not content:
            return None

        # Remove unwanted elements
        for element in content.select('script, style, nav, header, footer, .ad, .advertisement'):
            element.decompose()

        # Get text and clean it up
        text = content.get_text(separator='\n', strip=True)
        # Remove excessive whitespace
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        return text

    def _extract_image(self, soup: BeautifulSoup, domain: str) -> Optional[str]:
        """Extract the main image URL from the article."""
        # Try domain-specific rules first
        if 'techcrunch.com' in domain:
            # Try post thumbnail first
            selectors = [
                'img.attachment-post-thumbnail.size-post-thumbnail.wp-post-image',
                'img.wp-post-image',
                'img.featured-image',
                'img[class*="post-thumbnail"]'
            ]
            
            for selector in selectors:
                img = soup.select_one(selector)
                if img and img.get('src'):
                    # Skip author images
                    if 'author' not in img['src'].lower() and 'avatar' not in img['src'].lower():
                        return img['src']
            
            # Fallback to first large image in content
            for img in soup.select('main.template-content img'):
                if img.get('src') and ('wp-content' in img['src'] or 'techcrunch' in img['src']):
                    # Skip author images
                    if 'author' not in img['src'].lower() and 'avatar' not in img['src'].lower():
                        return img['src']
        else:
            # Generic image extraction
            # Try common image selectors
            selectors = [
                'meta[property="og:image"]',
                'meta[name="twitter:image"]',
                'img.featured-image',
                'img.article-image',
                'img[class*="hero"]',
                'img[class*="featured"]'
            ]
            
            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    if element.name == 'meta':
                        return element.get('content')
                    elif element.name == 'img':
                        return element.get('src')
            
            # Fallback to first large image in content
            for img in soup.select('article img, main img'):
                if img.get('src') and not any(x in img['src'].lower() for x in ['icon', 'logo', 'avatar', 'author']):
                    return img['src']
        
        return None

    def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch content from URL with error handling and retries."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def _extract_comments(self, url: str) -> Optional[str]:
        """Extract comments from a comments page."""
        if not url:
            return None

        content = self._fetch_url(url)
        if not content:
            return None

        soup = BeautifulSoup(content, 'html.parser')
        
        # Common comment container classes/IDs
        comment_selectors = [
            'div.comments', 'div.comment', 'div#comments',
            'section.comments', 'article.comment',
            'div.discussion', 'div#discussion',
            'div.feedback', 'div#feedback'
        ]
        
        comments_text = []
        for selector in comment_selectors:
            comments = soup.select(selector)
            for comment in comments:
                # Try to get the comment text, removing any nested comments
                comment_text = comment.get_text(strip=True, separator=' ')
                if comment_text:
                    comments_text.append(comment_text)
        
        if comments_text:
            return "\n\n".join(comments_text)
        return None

    def fetch_article_content(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch article content and extract main image."""
        content = self._fetch_url(url)
        if not content:
            return None, None

        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract main content
        article_text = ""
        
        # Try to find the main article content
        article_selectors = [
            'article', 'div.article', 'div.post', 'div.entry',
            'div.content', 'div.main', 'main'
        ]
        
        for selector in article_selectors:
            article = soup.select_one(selector)
            if article:
                article_text = article.get_text(strip=True, separator='\n')
                break
        
        if not article_text:
            # Fallback to body text if no article container found
            article_text = soup.body.get_text(strip=True, separator='\n')
        
        # Extract main image
        image_url = None
        image_selectors = [
            'meta[property="og:image"]',
            'meta[name="twitter:image"]',
            'meta[property="og:image:secure_url"]',
            'meta[property="og:image:url"]'
        ]
        
        for selector in image_selectors:
            img_tag = soup.select_one(selector)
            if img_tag and img_tag.get('content'):
                image_url = img_tag['content']
                break
        
        if not image_url:
            # Fallback to first large image in article
            for img in soup.find_all('img'):
                if img.get('src'):
                    src = img['src']
                    if not src.startswith(('data:', 'http')):
                        # Handle relative URLs
                        parsed_url = urlparse(url)
                        src = f"{parsed_url.scheme}://{parsed_url.netloc}{src}"
                    image_url = src
                    break
        
        return article_text, image_url

    def fetch_article_with_comments(self, article) -> Tuple[Optional[str], Optional[str]]:
        """Fetch article content and comments, combining them into a single text."""
        content, image_url = self.fetch_article_content(article.link)
        
        if content and article.comments_link:
            comments = self._extract_comments(article.comments_link)
            if comments:
                content += "\n\nCOMMENTS:\n" + comments
        
        return content, image_url 