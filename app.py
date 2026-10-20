import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from io import BytesIO

from pypdf import PdfReader
from docx import Document

from agent import build_agent

st.set_page_config(page_title="Customer Support AI", page_icon="💬", layout="wide")
st.title("💬 Customer Support AI Agent")
st.caption("CrewAI • Gemini 3.5 Flash-Lite • Excel order lookup • Human escalation")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_docs" not in st.session_state:
    st.session_state.knowledge_docs = []
if "tickets" not in st.session_state:
    st.session_state.tickets = []
if "agent" not in st.session_state:
    st.session_state.agent = None

with st.sidebar:
    st.header("Knowledge base")
    st.write("Upload company policies, FAQs, or product documents. No prebuilt embeddings or FAISS index is needed.")
    uploads = st.file_uploader(
        "Upload knowledge documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Process uploaded documents", use_container_width=True):
        docs = []
        if uploads:
            for uploaded in uploads:
                name = uploaded.name
                raw = uploaded.getvalue()
                try:
                    if name.lower().endswith(".pdf"):
                        reader = PdfReader(BytesIO(raw))
                        for page_num, page in enumerate(reader.pages, start=1):
                            text = page.extract_text() or ""
                            if text.strip():
                                docs.append({"source_file": name, "source_path": name,
                                             "page_number": page_num, "text": text})
                    elif name.lower().endswith(".docx"):
                        document = Document(BytesIO(raw))
                        text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
                        if text.strip():
                            docs.append({"source_file": name, "source_path": name,
                                         "page_number": None, "text": text})
                    else:
                        text = raw.decode("utf-8", errors="replace")
                        if text.strip():
                            docs.append({"source_file": name, "source_path": name,
                                         "page_number": None, "text": text})
                except Exception as exc:
                    st.error(f"Could not read {name}: {exc}")
        st.session_state.knowledge_docs = docs
        st.success(f"Processed {len(docs)} document section(s).")
    st.caption(f"Knowledge sections loaded: {len(st.session_state.knowledge_docs)}")

    st.divider()
    st.header("Order database")
    st.write("The app reads `orders.xlsx` from the repository.")
    if os.path.exists("orders.xlsx"):
        st.success("orders.xlsx found")
    else:
        st.warning("Upload orders.xlsx to the GitHub repository.")

    st.divider()
    st.header("Pending human tickets")
    st.metric("Pending tickets", len([t for t in st.session_state.tickets if t.get("status") == "Pending"]))
    if st.session_state.tickets:
        for ticket in reversed(st.session_state.tickets):
            with st.expander(f"{ticket['ticket_id']} — {ticket['status']}"):
                st.write(ticket["summary"])
                st.caption(f"Created: {ticket['created_at']}")
        ticket_json = json.dumps(st.session_state.tickets, indent=2)
        st.download_button("Download ticket records (JSON)", ticket_json, "pending_tickets.json",
                           mime="application/json", use_container_width=True)
    else:
        st.info("No tickets have been created in this session.")

    if st.button("Clear chat and session tickets", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tickets = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about a company policy or an order…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("GEMINI_API_KEY is missing. Add it in Streamlit Cloud → Settings → Secrets.")
            st.stop()

        if st.session_state.agent is None:
            st.session_state.agent = build_agent(
                api_key=st.secrets["GEMINI_API_KEY"],
                knowledge_docs=st.session_state.knowledge_docs,
                tickets=st.session_state.tickets,
            )
        else:
            # Refresh tool data so newly uploaded documents and tickets are available.
            st.session_state.agent = build_agent(
                api_key=st.secrets["GEMINI_API_KEY"],
                knowledge_docs=st.session_state.knowledge_docs,
                tickets=st.session_state.tickets,
            )

        history = st.session_state.messages[:-1][-12:]
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        ) or "No earlier messages."

        task_input = (
            f"Conversation so far (use this to resolve references like 'that order'):\n{history_text}\n\n"
            f"Latest customer message:\n{prompt}\n\n"
            "Answer using the tools when needed. Never invent policies or order records. "
            "If the customer requests a human or the issue cannot be resolved, create a ticket."
        )

        with st.chat_message("assistant"):
            with st.spinner("Checking the available information..."):
                result = st.session_state.agent.kickoff({"customer_request": task_input})
                answer = str(result)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as exc:
        st.error(
            "The agent could not complete this request. Check the Streamlit app logs and confirm "
            "your API key, model access, and package installation.\n\n"
            f"Technical detail: {exc}"
        )

st.caption("Demo note: session chat and tickets are held in Streamlit session memory. For durable tickets across restarts, connect a persistent database.")
