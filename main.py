from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from requests.auth import HTTPBasicAuth
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import uvicorn

# === CONFIGURATION ===
SERVICENOW_INSTANCE = "https://dev275692.service-now.com"
USERNAME = "admin"
PASSWORD = "-DJ@Xz11ruLy"
TABLE = "incident"
TOP_K = 3

# === LOAD MODEL ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === INIT APP ===
app = FastAPI()

# === INPUT SCHEMA ===
class Query(BaseModel):
    query: str
    ticket_sys_id: str  # sys_id of the current ticket to update work_notes

# === FETCH ALL TICKETS ===
def fetch_all_tickets():
    url = f"{SERVICENOW_INSTANCE}/api/now/table/{TABLE}?sysparm_limit=100"
    headers = {"Accept": "application/json"}
    auth = HTTPBasicAuth(USERNAME, PASSWORD)

    response = requests.get(url, headers=headers, auth=auth)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch tickets")

    records = response.json().get("result", [])
    tickets = []
    for rec in records:
        tickets.append({
            "number": rec.get("number", ""),
            "short_description": rec.get("short_description", ""),
            "description": rec.get("description", ""),
            "work_notes": rec.get("work_notes", ""),
            "resolution_notes": rec.get("resolution_notes", ""),
            "sys_id": rec.get("sys_id", "")
        })
    return tickets

# === BUILD FAISS INDEX ===
def build_faiss_index(tickets):
    texts = [
        f"{t['short_description']} {t['description']} {t['work_notes']} {t['resolution_notes']}"
        for t in tickets
    ]
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# === SUMMARY GENERATOR ===
def generate_summary(ticket):
    issue = ticket["short_description"] or ticket["description"] or "Not mentioned"
    actions = ticket["work_notes"] or "No actions recorded"
    resolution = ticket["resolution_notes"] or "Resolution not documented"

    summary = (
        f"- Issue: {issue.strip()}\n"
        f"- Actions: {actions.strip()}\n"
        f"- Resolution: {resolution.strip()}"
    )
    return summary

# === APPEND SUMMARY TO WORK_NOTES ===
def append_to_work_notes(ticket_sys_id, note):
    url = f"{SERVICENOW_INSTANCE}/api/now/table/{TABLE}/{ticket_sys_id}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    auth = HTTPBasicAuth(USERNAME, PASSWORD)

    payload = {
        "work_notes": f"\n\nSuggested Resolutions:\n{note}"
    }

    response = requests.patch(url, json=payload, headers=headers, auth=auth)
    if response.status_code not in [200, 204]:
        raise HTTPException(status_code=500, detail="Failed to update work_notes")

# === MAIN ENDPOINT ===
@app.post("/suggest_resolution")
def suggest_resolution(query: Query):
    tickets = fetch_all_tickets()
    if not tickets:
        raise HTTPException(status_code=404, detail="No tickets found")

    index, embeddings = build_faiss_index(tickets)
    query_embedding = model.encode([query.query])
    D, I = index.search(np.array(query_embedding), TOP_K)

    results = []
    all_summaries = []

    for rank, i in enumerate(I[0]):
        ticket = tickets[i]
        score = float(D[0][rank])
        summary = generate_summary(ticket)
        results.append({
            "ticket_number": ticket["number"],
            "similarity_score": round(score, 2),
            "summary": summary
        })
        all_summaries.append(f"{rank+1}. Ticket {ticket['number']} (Score: {round(score, 2)}):\n{summary}")

    # Combine summaries and update current ticket's work_notes
    final_notes = "\n\n".join(all_summaries)
    append_to_work_notes(query.ticket_sys_id, final_notes)

    return {"response": results}

# === RUN APP ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
