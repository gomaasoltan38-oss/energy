"""
EnergyGrammar — FastAPI Backend
================================
Run:
  pip install fastapi uvicorn langchain langchain-groq neo4j python-dotenv pydantic httpx
  uvicorn main:app --reload --port 8000

Env (.env):
  GROQ_API_KEY=gsk_...
  NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=...
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="EnergyGrammar API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Neo4j Connection ──────────────────────────────────────────────────────────
neo4j_driver = None

def get_neo4j_driver():
    global neo4j_driver
    if neo4j_driver is None:
        neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
    return neo4j_driver

# ── Groq LLM ──────────────────────────────────────────────────────────────────
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.1,
    )

# ── Extraction Prompt ─────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """
You are an energy-risk extraction system for facility maintenance logs.

IMPORTANT RULES:
1. Only extract information if the text describes a REAL maintenance issue or equipment problem.
2. If the text is random, nonsensical, offensive, or not related to facility maintenance, return:
   {{"valid": false, "reason": "not_maintenance_text"}}
3. If it IS a valid maintenance ticket, return:
   {{"valid": true, "component": "...", "symptom": "...", "location": "...", "energy_risk_keywords": [...]}}

Examples of INVALID text: "انا غبي", "hello world", "test", "asdfgh", "مرحبا"
Examples of VALID text: "المكيف ينقط ماء", "boiler making noise", "water leak in room 204"

Ticket text: {ticket_text}

Output ONLY valid JSON, nothing else.
"""

# ── Risk Score Algorithm ──────────────────────────────────────────────────────
def calculate_risk_score(entities: dict, similar_patterns: int) -> int:
    score = 0
    score += min(similar_patterns * 20, 40)
    score += min(len(entities.get("energy_risk_keywords", [])) * 15, 30)
    critical = ["boiler", "hvac", "chiller", "compressor", "steam", "مكيف", "غلاية", "ضاغط"]
    comp = entities.get("component", "").lower()
    if any(c in comp for c in critical):
        score += 20
    return min(score, 100)

def get_risk_level(score: int) -> str:
    if score >= 66: return "HIGH"
    if score >= 31: return "MEDIUM"
    return "LOW"

# ── Neo4j Operations ──────────────────────────────────────────────────────────
def save_ticket_to_neo4j(driver, ticket: dict):
    with driver.session() as session:
        # Create Ticket node
        session.run("""
            MERGE (t:Ticket {id: $id})
            SET t.text = $text,
                t.risk_score = $risk_score,
                t.risk_level = $risk_level,
                t.date = $date,
                t.resolved = false,
                t.component = $component,
                t.symptom = $symptom,
                t.location = $location,
                t.recommended_action = $recommended_action
        """, **ticket)

        # Component
        session.run("""
            MERGE (c:Component {name: $component})
            WITH c
            MATCH (t:Ticket {id: $ticket_id})
            MERGE (t)-[:MENTIONS]->(c)
        """, component=ticket["component"], ticket_id=ticket["id"])

        # Symptom + relationship from Component
        session.run("""
            MERGE (s:Symptom {name: $symptom})
            WITH s
            MATCH (c:Component {name: $component})
            MERGE (c)-[:HAS_SYMPTOM]->(s)
        """, symptom=ticket["symptom"], component=ticket["component"])

        # Location
        session.run("""
            MERGE (l:Location {name: $location})
            WITH l
            MATCH (t:Ticket {id: $ticket_id})
            MERGE (t)-[:LOCATED_AT]->(l)
        """, location=ticket["location"], ticket_id=ticket["id"])

        # Risk indicator
        session.run("""
            MATCH (s:Symptom {name: $symptom})
            MATCH (t:Ticket {id: $ticket_id})
            MERGE (s)-[:INDICATES_RISK {weight: $risk_score}]->(t)
        """, symptom=ticket["symptom"], ticket_id=ticket["id"], risk_score=ticket["risk_score"])

def get_similar_patterns(driver, component: str) -> int:
    with driver.session() as session:
        result = session.run("""
            MATCH (t:Ticket)-[:MENTIONS]->(c:Component {name: $component})
            RETURN count(t) as count
        """, component=component)
        record = result.single()
        return record["count"] if record else 0

# ── Pydantic Models ───────────────────────────────────────────────────────────
class TicketRequest(BaseModel):
    text: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("MATCH (t:Ticket) RETURN count(t) as count")
            count = result.single()["count"]
        neo4j_status = "connected"
    except Exception as e:
        count = 0
        neo4j_status = f"error: {str(e)[:50]}"
    return {"neo4j": neo4j_status, "groq": "ready", "total_tickets": count}


@app.post("/api/analyze-ticket")
async def analyze_ticket(request: TicketRequest):
    text = request.text.strip()

    # Basic validation
    if len(text) < 10:
        return {"valid": False, "error": "Text too short", "risk_score": 0, "risk_level": "INVALID"}

    # LLM extraction
    try:
        llm = get_llm()
        prompt = PromptTemplate(template=EXTRACTION_PROMPT, input_variables=["ticket_text"])
        chain = LLMChain(llm=llm, prompt=prompt)
        raw = chain.run(ticket_text=text)

        # Parse JSON
        clean = raw.strip().strip("```json").strip("```").strip()
        extracted = json.loads(clean)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # LLM validity check
    if not extracted.get("valid", True):
        return {"valid": False, "error": "Not a valid maintenance ticket", "risk_score": 0, "risk_level": "INVALID"}

    # Risk scoring
    driver = get_neo4j_driver()
    similar = get_similar_patterns(driver, extracted.get("component", ""))
    risk_score = calculate_risk_score(extracted, similar)
    risk_level = get_risk_level(risk_score)

    ticket = {
        "id": f"TKT-{str(uuid.uuid4())[:8].upper()}",
        "text": text,
        "component": extracted.get("component", "General Equipment"),
        "symptom": extracted.get("symptom", "General Issue"),
        "location": extracted.get("location", "Main Building"),
        "energy_risk_keywords": extracted.get("energy_risk_keywords", []),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "similar_patterns": similar,
        "recommended_action": get_recommended_action(extracted, risk_level),
        "date": datetime.now().isoformat(),
        "resolved": False,
    }

    # Save to Neo4j
    save_ticket_to_neo4j(driver, ticket)

    return {"valid": True, "ticket": ticket}


@app.get("/api/tickets")
async def get_tickets():
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (t:Ticket)
            RETURN t ORDER BY t.date DESC
        """)
        return [dict(r["t"]) for r in result]


@app.get("/api/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run("MATCH (t:Ticket {id: $id}) RETURN t", id=ticket_id)
        record = result.single()
        if not record:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return dict(record["t"])


@app.get("/api/graph-data")
async def get_graph_data():
    driver = get_neo4j_driver()
    nodes = []
    edges = []

    with driver.session() as session:
        # All nodes
        result = session.run("""
            MATCH (n)
            RETURN labels(n) AS labels, n.name AS name, n.id AS id,
                   n.risk_score AS risk_score, n.risk_level AS risk_level
        """)
        for r in result:
            label = r["labels"][0] if r["labels"] else "Unknown"
            node_id = r["id"] or r["name"] or "unknown"
            nodes.append({
                "id": node_id,
                "label": r["name"] or r["id"] or "?",
                "type": label,
                "risk_score": r["risk_score"],
                "risk_level": r["risk_level"],
            })

        # All edges
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN a.id AS source_id, a.name AS source_name,
                   b.id AS target_id, b.name AS target_name,
                   type(r) AS relationship
        """)
        for r in result:
            src = r["source_id"] or r["source_name"]
            tgt = r["target_id"] or r["target_name"]
            if src and tgt:
                edges.append({"source": src, "target": tgt, "relationship": r["relationship"]})

    return {"nodes": nodes, "edges": edges}


@app.get("/api/stats")
async def get_stats():
    driver = get_neo4j_driver()
    with driver.session() as session:
        counts = session.run("""
            MATCH (t:Ticket)
            RETURN count(t) AS total,
                   sum(CASE WHEN t.risk_level = 'HIGH' THEN 1 ELSE 0 END) AS high,
                   sum(CASE WHEN t.risk_level = 'MEDIUM' THEN 1 ELSE 0 END) AS medium,
                   sum(CASE WHEN t.risk_level = 'LOW' THEN 1 ELSE 0 END) AS low
        """).single()

        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]

    return {
        "total_tickets": counts["total"],
        "high_risk_count": counts["high"],
        "energy_savings_pct": 32,
        "active_nodes": node_count,
    }


@app.get("/api/alerts")
async def get_alerts():
    """Returns ALL HIGH risk tickets — queried live from Neo4j"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (t:Ticket)
            WHERE t.risk_level = 'HIGH'
            RETURN t ORDER BY t.risk_score DESC
        """)
        return [dict(r["t"]) for r in result]


@app.patch("/api/alerts/{ticket_id}/resolve")
async def resolve_alert(ticket_id: str):
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run("""
            MATCH (t:Ticket {id: $id})
            SET t.resolved = true
        """, id=ticket_id)
    return {"success": True}


# ── Seed Data ─────────────────────────────────────────────────────────────────
SEED_TICKETS = [
    {"text": "المكيف في الطابق الثالث ينقط ماء وفيه صوت غريب من الكمبروسر",
     "component": "HVAC / Compressor", "symptom": "Water Dripping + Abnormal Noise",
     "location": "Third Floor", "risk_score": 82},
    {"text": "Water leaking from ceiling vent in room 204, unusual humidity",
     "component": "Ceiling Vent", "symptom": "Water Leak + High Humidity",
     "location": "Room 204", "risk_score": 74},
    {"text": "Boiler pressure dropping intermittently, slight rust on intake pipe",
     "component": "Boiler", "symptom": "Pressure Drop + Rust",
     "location": "Boiler Room", "risk_score": 91},
    {"text": "تسريب مياه تبريد تحت وحدة الـ chiller في سطح المبنى",
     "component": "Chiller Unit", "symptom": "Coolant Leak",
     "location": "Rooftop", "risk_score": 85},
    {"text": "HVAC unit making rattling noise, filter not changed in 6 months",
     "component": "HVAC Unit", "symptom": "Rattling Noise + Dirty Filter",
     "location": "Floor 2", "risk_score": 55},
]


@app.on_event("startup")
async def seed_database():
    """Seed Neo4j with initial tickets on startup"""
    try:
        driver = get_neo4j_driver()
        for i, t in enumerate(SEED_TICKETS):
            ticket = {
                "id": f"TKT-{str(i+1).zfill(3)}",
                "text": t["text"],
                "component": t["component"],
                "symptom": t["symptom"],
                "location": t["location"],
                "energy_risk_keywords": ["hvac", "leak", "energy"],
                "risk_score": t["risk_score"],
                "risk_level": get_risk_level(t["risk_score"]),
                "similar_patterns": 2,
                "recommended_action": "Inspect and service immediately.",
                "date": datetime.now().isoformat(),
                "resolved": False,
            }
            save_ticket_to_neo4j(driver, ticket)
    except Exception as e:
        print(f"Seed warning: {e}")


def get_recommended_action(entities: dict, risk_level: str) -> str:
    comp = entities.get("component", "").lower()
    if "boiler" in comp: return "Critical: Schedule boiler inspection immediately. Check pressure valves."
    if "chiller" in comp: return "Isolate chiller unit. Inspect refrigerant lines."
    if "hvac" in comp or "compressor" in comp: return "Inspect compressor seals. Replace air filter."
    if "steam" in comp: return "Re-insulate steam lines. High heat loss risk."
    if "pump" in comp: return "Inspect pump motor and impeller."
    if risk_level == "HIGH": return "Immediate inspection required."
    if risk_level == "MEDIUM": return "Schedule maintenance within 48 hours."
    return "Log for routine maintenance."


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
