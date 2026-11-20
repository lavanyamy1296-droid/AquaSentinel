from datetime import datetime
from io import BytesIO
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = FastAPI(title="AquaSentinel AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
FEATURES = ["ph", "turbidity_ntu", "dissolved_oxygen_mg_l", "tds_mg_l", "temperature_c"]

def risk_label(r):
    points = (2 if r["ph"] < 6.5 or r["ph"] > 8.5 else 0)
    points += (2 if r["turbidity_ntu"] > 5 else 0)
    points += (2 if r["dissolved_oxygen_mg_l"] < 5 else 0)
    points += (1 if r["tds_mg_l"] > 500 else 0)
    points += (1 if r["temperature_c"] > 32 else 0)
    return "High" if points >= 4 else ("Moderate" if points >= 2 else "Low")

def make_data():
    rng = np.random.default_rng(17)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=180)
    rows = []
    for i, date in enumerate(dates):
        r = {"date": date,
             "ph": round(float(rng.normal(7.35 + .15*np.sin(i/18), .28)), 2),
             "turbidity_ntu": round(float(max(.2, rng.normal(2 + .45*np.sin(i/13), .65))), 2),
             "dissolved_oxygen_mg_l": round(float(max(2, rng.normal(7.1 - .25*np.sin(i/20), .75))), 2),
             "tds_mg_l": round(float(max(100, rng.normal(320 + 20*np.sin(i/24), 45))), 1),
             "temperature_c": round(float(rng.normal(27 + 2*np.sin(i/30), 2)), 1)}
        if i in (34, 89, 142):
            r["turbidity_ntu"] = round(r["turbidity_ntu"]*3.5, 2)
            r["dissolved_oxygen_mg_l"] = round(r["dissolved_oxygen_mg_l"]*.55, 2)
            r["tds_mg_l"] = round(r["tds_mg_l"]*1.25, 1)
        r["risk"] = risk_label(r)
        rows.append(r)
    return pd.DataFrame(rows)

DATA = make_data()
MODEL = RandomForestClassifier(n_estimators=220, random_state=17, class_weight="balanced").fit(DATA[FEATURES], DATA["risk"])

def quality_score(r):
    loss = min(25, abs(float(r["ph"])-7.2)*10)
    loss += min(25, max(0, float(r["turbidity_ntu"])-1)*3.5)
    loss += min(25, max(0, 6.5-float(r["dissolved_oxygen_mg_l"]))*5)
    loss += min(15, max(0, float(r["tds_mg_l"])-300)*.04)
    loss += min(10, max(0, float(r["temperature_c"])-28)*1.5)
    return int(max(0, min(100, round(100-loss))))

def dashboard_data():
    flags = IsolationForest(contamination=.035, random_state=17).fit_predict(DATA[FEATURES])
    anomalies = []
    for row, flag in zip(DATA.to_dict("records"), flags):
        if flag == -1:
            anomalies.append({k: (v.strftime("%Y-%m-%d") if k == "date" else v) for k,v in row.items()})
    cur = DATA.iloc[-1].to_dict()
    cur = {k: (v.strftime("%Y-%m-%d") if k == "date" else v) for k,v in cur.items()}
    cur["quality_score"] = quality_score(cur)
    cur["predicted_risk"] = str(MODEL.predict(pd.DataFrame([cur])[FEATURES])[0])
    history = []
    for row in DATA.tail(45).to_dict("records"):
        item = {k: (v.strftime("%Y-%m-%d") if k == "date" else v) for k,v in row.items()}
        item["quality_score"] = quality_score(row)
        history.append(item)
    counts = DATA["risk"].value_counts()
    return {"current": cur, "history": history, "anomalies": anomalies[-8:],
            "risk_distribution": [{"risk": k, "count": int(counts.get(k,0))} for k in ["Low","Moderate","High"]],
            "summary": {"average_ph": round(float(DATA.ph.mean()),2),
                        "average_turbidity": round(float(DATA.turbidity_ntu.mean()),2),
                        "average_oxygen": round(float(DATA.dissolved_oxygen_mg_l.mean()),2),
                        "anomaly_count": len(anomalies)}}

@app.get("/")
def home():
    return {"status":"ok","name":"AquaSentinel AI"}

@app.get("/api/health")
def health():
    return {"status":"healthy"}

@app.get("/api/dashboard")
def dashboard():
    return dashboard_data()

@app.get("/api/analyze")
def analyze(ph: float, turbidity_ntu: float, dissolved_oxygen_mg_l: float, tds_mg_l: float, temperature_c: float):
    r = {"ph":ph, "turbidity_ntu":turbidity_ntu, "dissolved_oxygen_mg_l":dissolved_oxygen_mg_l, "tds_mg_l":tds_mg_l, "temperature_c":temperature_c}
    limits = {"ph":(0,14), "turbidity_ntu":(0,1000), "dissolved_oxygen_mg_l":(0,30), "tds_mg_l":(0,10000), "temperature_c":(-5,100)}
    for k,(lo,hi) in limits.items():
        if not lo <= r[k] <= hi:
            raise HTTPException(422, f"{k} must be between {lo} and {hi}")
    pred = str(MODEL.predict(pd.DataFrame([r])[FEATURES])[0])
    notes = []
    if ph < 6.5 or ph > 8.5: notes.append("pH is outside the illustrative demo band (6.5–8.5).")
    if turbidity_ntu > 5: notes.append("Turbidity is elevated in this demo screening.")
    if dissolved_oxygen_mg_l < 5: notes.append("Dissolved oxygen is below the demo reference.")
    if tds_mg_l > 500: notes.append("TDS is above the demo reference of 500 mg/L.")
    if temperature_c > 32: notes.append("Temperature is high relative to the demo reference.")
    if not notes: notes.append("No parameter crossed the configured demo screening thresholds.")
    return {"predicted_risk":pred, "quality_score":quality_score(r), "observations":notes,
            "model_note":"Random Forest trained on synthetic demo data; not a laboratory test or safety certification."}

@app.get("/api/recommendations")
def recommendations():
    return {"recommendations":[
        {"title":"Validate sensor readings","detail":"Calibrate probes according to the sensor manufacturer's instructions.","priority":"Routine"},
        {"title":"Investigate turbidity changes","detail":"Inspect filtration and possible suspended-particle sources if readings rise.","priority":"High"},
        {"title":"Maintain sampling history","detail":"Compare repeated measurements and investigate persistent changes.","priority":"Routine"},
        {"title":"Confirm with a laboratory","detail":"Use accredited laboratory testing for real water-quality decisions.","priority":"Essential"}]}

@app.post("/api/report/pdf")
def report_pdf():
    d = dashboard_data()
    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    y = 790
    pdf.setTitle("AquaSentinel AI Report")
    pdf.setFont("Helvetica-Bold",18)
    pdf.drawString(45,y,"AquaSentinel AI - Monitoring Report")
    y -= 32
    pdf.setFont("Helvetica",10)
    lines = [f"Generated: {datetime.now():%d %b %Y, %H:%M}",
             f"Latest reading: {d['current']['date']}",
             f"Illustrative score: {d['current']['quality_score']}/100",
             f"Model risk category: {d['current']['predicted_risk']}",
             f"pH: {d['current']['ph']} | Turbidity: {d['current']['turbidity_ntu']} NTU",
             f"Dissolved oxygen: {d['current']['dissolved_oxygen_mg_l']} mg/L",
             f"TDS: {d['current']['tds_mg_l']} mg/L | Temperature: {d['current']['temperature_c']} C",
             "", "IMPORTANT: Synthetic demo data and illustrative thresholds only.",
             "Not a drinking-water safety determination or substitute for accredited lab analysis."]
    for line in lines:
        pdf.drawString(45,y,line)
        y -= 21
    pdf.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition":"attachment; filename=aquasentinel-report.pdf"})
