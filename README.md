# Customer Support AI Agent

A beginner-friendly customer support demo built with Streamlit, CrewAI, Gemini 3.5 Flash-Lite, and an Excel order spreadsheet.

## Features

- One CrewAI agent
- Gemini API key read from Streamlit Secrets
- Upload PDF, DOCX, TXT, and Markdown company knowledge documents
- On-demand text chunking with metadata (no prebuilt embeddings, FAISS index, or chunks.json)
- Excel order lookup from `orders.xlsx`
- Session chat context
- Human escalation tickets shown as Pending
- Download pending ticket records as JSON

## Files

- `app.py`: Streamlit interface, document upload, chat history, ticket display
- `agent.py`: one CrewAI agent and three tools
- `orders.xlsx`: 60 synthetic demo orders
- `requirements.txt`: pinned dependencies

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files from this folder to the repository root.
3. Open Streamlit Community Cloud and create an app from that repository.
4. Set the main file path to `app.py`.
5. In the deployed app, open **Settings → Secrets** and add:

   ```toml
   GEMINI_API_KEY = "your_real_gemini_api_key"
   ```

6. Save and let Streamlit install `requirements.txt`.

Never commit your API key to GitHub.

## Important limitations

- The model ID used is `gemini/gemini-3.5-flash-lite` through CrewAI/LiteLLM. Access depends on your Google AI API account and model availability.
- Company knowledge is processed in the running session. Upload documents again after a new session.
- Search is simple keyword ranking, not semantic vector search. It is intended as a beginner baseline.
- The spreadsheet is a simulated database and does not provide live courier tracking.
- Chat and tickets are kept in Streamlit session state. Tickets are not durable across restarts, and no real human is notified. For production, connect persistent storage and a real helpdesk/email channel.
- The app has not been locally tested, in accordance with the requested workflow. If deployment fails, inspect the Streamlit Cloud logs for the exact dependency or API error.
