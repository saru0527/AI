from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from requests.auth import HTTPBasicAuth
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
import re

# OpenAI SDK (new)
from openai import OpenAI

# === CONFIGURATION ===
SERVICENOW_INSTANCE = "https://dev275692.service-now.com"
USERNAME = os.getenv("SN_USERNAME", "admin")
PASSWORD = os.getenv("SN_PASSWORD", "-DJ@Xz11ruLy")  # Store securely in production
TABLE = "incident"
TOP_K = 4

# === OpenAI Setup ===
# === Note: My personal open api key not cognizant api key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")  # Replace with your actual key or set as env var
client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = "gpt-3.5-turbo"  # use GPT-3.5 to avoid access issues

# === Load Embedding Model ===
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Init FastAPI ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Health Check ===
@app.get("/ping")
def ping():
    return {"message": "API is up and running "}

# === Input Schema ===
class Query(BaseModel):
    query: str
    ticket_sys_id: str

# === Mask Sensitive Data ===
def mask_sensitive(text):
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[EMAIL]', text)
    return re.sub(r'\b\d{6,}\b', '[NUMBER]', text)

# === Fetch Tickets from ServiceNow ===
def fetch_all_tickets():
    url = f"{SERVICENOW_INSTANCE}/api/now/table/{TABLE}?sysparm_limit=100&sysparm_fields=number,short_description,description,work_notes,resolution_notes,close_notes,u_resolution_notes,comments,sys_id"
    headers = {"Accept": "application/json"}
    auth = HTTPBasicAuth(USERNAME, PASSWORD)

    response = requests.get(url, headers=headers, auth=auth)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch tickets")

    records = response.json().get("result", [])
    seen_ids = set()
    tickets = []

    for rec in records:
        sys_id = rec.get("sys_id", "")
        if sys_id in seen_ids:
            continue
        seen_ids.add(sys_id)

        resolution_note = (
            rec.get("resolution_notes")
            or rec.get("close_notes")
            or rec.get("u_resolution_notes")
            or rec.get("comments")
            or ""
        )
        ticket = {
            "number": rec.get("number", ""),
            "short_description": rec.get("short_description", ""),
            "description": rec.get("description", ""),
            "work_notes": rec.get("work_notes", ""),
            "resolution_notes": resolution_note,
            "sys_id": sys_id
        }
        tickets.append(ticket)
    return tickets

# === FAISS Embedding Index ===
faiss_index = None
ticket_embeddings = None
ticket_cache = None

def build_faiss_index(tickets):
    global faiss_index, ticket_embeddings, ticket_cache

    if faiss_index and ticket_cache == tickets:
        return faiss_index, ticket_embeddings

    texts = [
        f"{t['short_description']} {t['description']} {t['work_notes']} {t['resolution_notes']}"
        for t in tickets
    ]
    embeddings = retriever_model.encode(texts)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))
    ticket_embeddings = embeddings
    ticket_cache = tickets
    return faiss_index, embeddings

# === Ticket Summary Builder ===
def generate_ticket_summary(ticket):
    issue = ticket["short_description"] or ticket["description"] or "Not mentioned"
    actions = ticket["work_notes"] or "No actions recorded"
    resolution = ticket["resolution_notes"] or "Resolution not documented"
    return f"- Issue: {mask_sensitive(issue.strip())}\n- Actions: {mask_sensitive(actions.strip())}\n- Resolution: {mask_sensitive(resolution.strip())}"

# === Prompt Builder ===
def build_chat_prompt(query, ticket_summaries):
    prompt = f"You are an IT support assistant helping resolve incidents using historical tickets.\n\nCurrent issue:\n\"{query}\"\n\nRelated tickets:\n"
    for idx, summary in enumerate(ticket_summaries, 1):
        prompt += f"\nTicket {idx}:\n{summary}\n"
    prompt += "\nBased on the above, provide a professional resolution to the current issue."
    return prompt

# === Query OpenAI GPT ===
def query_openai(prompt: str):
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an IT support assistant that provides clear resolutions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI model query failed: {str(e)}")

# === Update Work Notes in ServiceNow ===
def append_to_work_notes(ticket_sys_id, full_note):
    url = f"{SERVICENOW_INSTANCE}/api/now/table/{TABLE}/{ticket_sys_id}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    auth = HTTPBasicAuth(USERNAME, PASSWORD)
    payload = {"work_notes": full_note}
    response = requests.patch(url, json=payload, headers=headers, auth=auth)
    if response.status_code not in [200, 204]:
        raise HTTPException(status_code=500, detail="Failed to update ticket notes")

# === Main Endpoint ===
@app.post("/suggest_resolution")
def suggest_resolution(query: Query):
    try:
        tickets = fetch_all_tickets()
    except Exception:
        raise HTTPException(status_code=500, detail="ServiceNow fetch failed")

    if not tickets:
        raise HTTPException(status_code=404, detail="No tickets found")

    index, embeddings = build_faiss_index(tickets)
    query_embedding = retriever_model.encode([query.query])
    D, I = index.search(np.array(query_embedding), TOP_K)

    results = []
    summaries = []

    for rank, i in enumerate(I[0]):
        ticket = tickets[i]
        score = float(D[0][rank])
        summary = generate_ticket_summary(ticket)
        results.append({
            "ticket_number": ticket["number"],
            "similarity_score": round(score, 4),
            "summary": summary
        })
        summaries.append(summary)

    prompt = build_chat_prompt(query.query, summaries)
    ai_response = query_openai(prompt)

    extracted_resolutions = re.findall(r"[Rr]esolution: (.+)", ai_response)
    if not extracted_resolutions:
        extracted_resolutions = ["No useful resolution notes extracted."]

    note_text = f"""
 Suggested AI Response:
{ai_response}

 Related Tickets:
- {"\n- ".join([f"{t['ticket_number']} (Score: {t['similarity_score']})" for t in results])}

 Ticket Details:
{chr(10).join([t['summary'] for t in results])}
""".strip()

    try:
        append_to_work_notes(query.ticket_sys_id, note_text)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not update ticket notes")

    return {
        "related_tickets": [r["ticket_number"] for r in results],
        "summary": ai_response,
        "top_tickets": results,
        "extracted_resolutions": extracted_resolutions
    }

# === Run the App ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)