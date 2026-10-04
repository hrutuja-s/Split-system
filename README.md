# 🏠 FlatSplit — Shared Flat Expense Tracker with a Learning Categoriser

Flatmates log expenses as `60 onions` from their own phones, any time. When the flat decides to settle,
the app splits the total equally, says who pays whom (fewest payments), tracks each payment (partial
payments too), closes the cycle and archives it with a PDF report.

**Cognitive layer:** auto-categorisation (TF-IDF + logistic regression that keeps learning from the flat's
own entries and corrections), typo/outlier detection (median + MAD), spend forecast, fair-buyer suggestion.

## Project layout
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (Add, Settle, History, Insights) |
| `logic.py` | Parsing, settlement algorithm, anomaly detection, forecast |
| `classifier.py` | Item categoriser (ML) |
| `db.py` | SQLAlchemy tables and queries (SQLite locally, Postgres online) |
| `report.py` | PDF settlement report |
| `train_eval.py` | Classifier evaluation for your CA4 report |
| `tests/` | Unit tests (`python -m pytest tests`) |

## 1. Run on your PC
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. Data is stored in a local file `flatsplit.db`.
To test from your phone on the same Wi-Fi, open `http://<your-PC-IP>:8501`.

## 2. Deploy (free) so all three of you can use it anywhere
SQLite is wiped on free hosts, so use a free hosted Postgres.

1. **Database:** create a free project on [Neon](https://neon.tech) (or Supabase) and copy the connection string
   (`postgresql://user:password@host/dbname?sslmode=require`). Tables are created automatically on first run.
2. **Code:** push this folder to a GitHub repository (`.gitignore` already excludes secrets and the local DB).
3. **Host:** on [Streamlit Community Cloud](https://share.streamlit.io) choose *New app*, pick the repo, main file `app.py`.
4. **Secrets:** in *App settings → Secrets* paste:
   ```toml
   DATABASE_URL = "your Neon/Supabase connection string"
   FLAT_PASSCODE = "something only your flat knows"
   ```
5. Share the app link. On each phone open it and use *Add to Home Screen*. Choose your name once; it is
   remembered in the page link.

Note: the app shows other people's entries after a page refresh (pull down), not instantly.

## 3. Evaluate 
```bash
python train_eval.py
```
Reports (1) cross-validated accuracy on unseen words (hard case) and (2) accuracy on realistic entries
(plurals, typos, quantities). Replace the test list in `train_eval.py` with entries from your own flat for a
stronger result. Use `Fix or delete an entry → Correct category` in the app to teach the model your items.

## Known limitations
- Equal split only (no "I didn't use it" exclusions yet).
- Categoriser only knows words from its seed list plus your corrections; unknown items land in "Other".
- Free hosting apps sleep after inactivity; the first open may take ~30 seconds.
