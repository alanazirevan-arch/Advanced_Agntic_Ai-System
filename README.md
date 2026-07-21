# Multi-Agent Report Generator

This project upgrades the original single-flow notebook into a multi-agent workflow with these agents:

- Manager Agent: starts the workflow
- Research Agent: gathers information with Tavily
- Summary Agent: condenses the results
- Writing Agent: drafts the report
- Review Agent: polishes and improves the final output

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your API keys to the .env file.

3. Run the project:
   ```bash
   python3 main.py
   ```

## API Keys

- Groq: https://console.groq.com/keys
- Tavily: https://app.tavily.com/home

Put the keys in the .env file using the .env.example template.
