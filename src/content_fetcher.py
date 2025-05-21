import requests
from bs4 import BeautifulSoup
import logging
import time
from typing import Optional
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

    def fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch and parse article content from URL."""
        try:
            # Clean the URL before making the request
            clean_url = self._clean_url(url)
            
            if self.verbose:
                logger.info(f"Processing URL: {url}")
                logger.info(f"Cleaned URL: {clean_url}")
            
            # Rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.delay_between_requests:
                time.sleep(self.delay_between_requests - time_since_last_request)
            
            # Make request
            response = self.session.get(clean_url, timeout=10)
            response.raise_for_status()

            if self.verbose:
                logger.info(f"Response status: {response.status_code}")
                self._save_to_cache(clean_url, response.text)
            
            # Update last request time
            self.last_request_time = time.time()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract domain for specific rules
            domain = self._get_domain(clean_url)
            
            if self.verbose:
                logger.info(f"Domain: {domain}")
            
            # Extract content
            content = self._extract_content(soup, domain)
            
            if content:
                if self.verbose:
                    logger.info(f"Successfully extracted content from {clean_url} (length: {len(content)} chars)")
                # Cache the content if in verbose mode
                self._save_to_cache(clean_url, content)
                
                # Extract image URL
                image_url = self._extract_image(soup, domain)
                if image_url:
                    if self.verbose:
                        logger.info(f"Found image: {image_url}")
                    return content, image_url
                return content, None
            else:
                logger.warning(f"Could not extract content from {clean_url}")
                if self.verbose:
                    logger.info("Available HTML structure:")
                    for tag in soup.find_all(['article', 'main', '.post-content', '.article-content']):
                        logger.info(f"Found potential content container: {tag.name} with class {tag.get('class', 'no-class')}")
                return None, None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {str(e)}")
            return None, None 