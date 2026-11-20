# AquaSentinel AI 💧

AI-assisted water-parameter monitoring and early-warning **demonstration** application. It visualizes synthetic readings, screens user-entered measurements, detects unusual patterns and creates a PDF summary.

## Features
- React dashboard with recent pH and turbidity trends
- Random Forest risk-category screening
- Isolation Forest anomaly detection
- Illustrative quality score and parameter observations
- Monitoring recommendations
- PDF monitoring report
- Manual sample analysis form

## Stack
Frontend: React, Vite, Recharts, Axios, Lucide  
Backend: FastAPI, pandas, NumPy, scikit-learn, ReportLab

## Requirements
Python 3.10+ and Node.js 18+.

## Run backend (Terminal 1)
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```
API: http://127.0.0.1:8000  
Docs: http://127.0.0.1:8000/docs

## Run frontend (Terminal 2)
```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```
App: http://localhost:5173

Keep both terminals open. `npm.cmd` avoids common PowerShell script-policy issues.

## API
- `GET /api/health`
- `GET /api/dashboard`
- `GET /api/analyze?ph=7.2&turbidity_ntu=2&dissolved_oxygen_mg_l=7&tds_mg_l=320&temperature_c=27`
- `GET /api/recommendations`
- `POST /api/report/pdf`

## Scientific limitation
The included dataset is synthetic and the thresholds are illustrative. Model output and the score are educational indicators only, not a drinking-water safety determination or regulatory compliance assessment. Do not use this demo to decide whether water is safe to drink. Real decisions require appropriate sampling, applicable standards and accredited laboratory analysis.

## GitHub
Create an empty repository named `AquaSentinel-AI`, then run from the project root:
```powershell
git init
git add .
git commit -m "Initial commit - AquaSentinel AI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AquaSentinel-AI.git
git push -u origin main
```
