# Daily Reading List Generator

This system automatically generates a daily reading list from configured RSS feeds, filtering and summarizing articles based on your interests.

## Features

- Fetches articles from multiple RSS feeds
- Filters articles based on keywords
- Generates summaries and insights using Google's Gemini AI
- Outputs a nicely formatted markdown file

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
4. Configure your RSS feeds and keywords in `config.json`

## Usage

Run the script:
```bash
python src/main.py
```

The script will:
1. Fetch articles from configured RSS feeds
2. Filter articles based on keywords
3. Generate summaries and insights using Gemini
4. Create a markdown file in the output directory

## Configuration

Edit `config.json` to:
- Add/remove RSS feeds
- Modify keywords for filtering
- Change the output directory

## Extending

The system is designed to be modular. You can:
- Add new content sources by implementing the `ContentSource` interface
- Create new filters by implementing the `ArticleFilter` interface
- Modify the markdown template in `MarkdownGenerator`
- Add new processing steps in `main.py`
