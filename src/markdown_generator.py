from typing import List
from datetime import datetime
from llm_processor import ProcessedArticle
import logging
import re

logger = logging.getLogger(__name__)

class MarkdownGenerator:
    def __init__(self, title: str = "Daily Reading List"):
        self.title = title
        self.template = """# Daily Reading List - {date}

{content}

## Articles

{articles}
"""

    def _normalize_headers(self, content: str) -> str:
        """Intelligently normalize headers based on their hierarchy."""
        # Split into lines
        lines = content.split('\n')
        normalized_lines = []
        
        # First pass: collect header information
        header_levels = []
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                header_levels.append(level)
        
        # Determine if we need to shift h2s to h4s
        has_h1 = 1 in header_levels
        has_h2 = 2 in header_levels
        needs_shift = has_h1 and has_h2
        
        # Second pass: normalize headers
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                content = header_match.group(2)
                
                if level == 1:
                    # Convert h1 to h3
                    normalized_lines.append(f"### {content}")
                elif level == 2 and needs_shift:
                    # Shift h2 to h4 if we have both h1 and h2
                    normalized_lines.append(f"#### {content}")
                else:
                    # Keep other headers as is, but cap at h4
                    new_level = min(4, level + 1)  # Shift everything up by 1
                    normalized_lines.append(f"{'#' * new_level} {content}")
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)

    def generate_markdown(self, processed_articles: List[ProcessedArticle]) -> str:
        """Generate markdown content from processed articles."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Start with header
        markdown = f"# {self.title} - {today}\n\n"
        
        # Add each article
        for article in processed_articles:
            markdown += f"## {article.article.title}\n\n"
            markdown += f"*Source: {article.article.source}*\n\n"
            
            if article.summary:
                # Normalize headers in the summary
                normalized_summary = self._normalize_headers(article.summary)
                markdown += f"{normalized_summary}\n\n"
            
            if article.why_care:
                markdown += f"### Why Should I Care?\n\n"
                markdown += f"{article.why_care}\n\n"
            
            # if article.insights:
            #     markdown += "### Key Insights\n\n"
            #     for insight in article.insights:
            #         markdown += f"- {insight}\n"
            #     markdown += "\n"
            
            markdown += f"[Read more]({article.article.link})\n\n"
            
            # Add related articles if any
            if article.article.related_links:
                markdown += "### Related Articles\n\n"
                for related in article.article.related_links:
                    markdown += f"- [{related.title}]({related.link})\n"
                markdown += "\n"
            
            markdown += "---\n\n"
        
        return markdown

    def save_markdown(self, content: str, filename: str):
        try:
            with open(filename, 'w') as f:
                f.write(content)
            logger.info(f"Successfully saved markdown to {filename}")
        except Exception as e:
            logger.error(f"Error saving markdown to {filename}: {str(e)}") 