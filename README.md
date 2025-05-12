# YouTube Content Enhancer

This application uses LangGraph to transform YouTube video transcripts into various content formats. It processes videos through a workflow that includes transcript extraction, content enhancement, formatting, and verification.

## Features

- Extract transcripts from YouTube videos
- Enhance and structure the content
- Convert content into multiple formats (blog post, LinkedIn post, Twitter thread)
- Verify content accuracy
- Modern Streamlit UI for easy interaction

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser to the provided URL (usually http://localhost:8501)
3. Enter a YouTube URL and click "Process Video"
4. View the generated content in different formats

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for YouTube access

## Note

This application uses the `youtube-transcript-api` package to fetch transcripts without requiring a YouTube API key. However, some videos may not have available transcripts or may have disabled transcript access. 