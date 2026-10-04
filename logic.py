"""Pure business logic: parsing, settlement, anomaly detection, forecasting.
No Streamlit or database code here, so it is easy to unit test."""
import re
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))

_START = re.compile(r"^\s*(?:₹|rs\.?\s*)?\d", re.I)
_END = re.compile(r"\d\s*$")
_P1 = re.compile(r"^\s*(?:₹|rs\.?\s*)?(\d+(?:\.\d{1,2})?)\s+(.+?)\s*$", re.I)
_P2 = re.compile(r"^\s*(.+?)\s+(?:₹|rs\.?\s*)?(\d+(?:\.\d{1,2})?)\s*$", re.I)


def parse_entry(text: str):
    """'60 onions' or 'onions 60' -> (60.0, 'onions'). Returns None if unclear."""
    t = (text or "").strip()
    if not t:
        return None
    if _START.match(t) and _END.search(t) and len(t.split()) > 2:
        return None  # e.g. '2 kg onions 60' is ambiguous
    m = _P1.match(t)
    if m:
        amount, item = float(m.group(1)), m.group(2).strip()
    else:
        m = _P2.match(t)
        if not m:
            return None
        amount, item = float(m.group(2)), m.group(1).strip()
    if amount <= 0 or not item or item.replace(".", "").isdigit():
        return None
    return amount, item


def settle(paid_by: dict, people: list):
    """Equal split. Returns totals plus the smallest practical list of payments
    (greedy: biggest debtor pays biggest creditor), at most n-1 transfers."""
    names = list(dict.fromkeys(list(people) + list(paid_by)))
    paise = {n: int(round(paid_by.get(n, 0) * 100)) for n in names}
    total = sum(paise.values())
    share = total / len(names) if names else 0
    bal = {n: int(round(paise[n] - share)) for n in names}
    cred = sorted([[n, b] for n, b in bal.items() if b > 0], key=lambda x: -x[1])
    debt = sorted([[n, -b] for n, b in bal.items() if b < 0], key=lambda x: -x[1])
    transfers, i, j = [], 0, 0
    while i < len(cred) and j < len(debt):
        a = min(cred[i][1], debt[j][1])
        transfers.append({"from": debt[j][0], "to": cred[i][0], "amount": a / 100})
        cred[i][1] -= a
        debt[j][1] -= a
        if cred[i][1] == 0:
            i += 1
        if debt[j][1] == 0:
            j += 1
    return {
        "names": names,
        "paid": {n: paise[n] / 100 for n in names},
        "total": total / 100,
        "share": share / 100,
        "transfers": transfers,
    }


def is_anomaly(amount: float, category: str, history: pd.DataFrame) -> bool:
    """Robust outlier check (median + MAD) against past spending in the same category."""
    peers = []
    if history is not None and len(history):
        peers = history.loc[history["category"] == category, "amount"].astype(float).tolist()
    if len(peers) < 4:
        return amount >= 2000
    med = float(np.median(peers))
    mad = float(np.median([abs(x - med) for x in peers])) or med * 0.3 or 1.0
    return amount > med + 4 * mad and amount > 2 * med


def forecast_month(entries: pd.DataFrame):
    """Projected 30-day household spend from the observed daily rate. Needs >= 7 days."""
    if entries is None or entries.empty:
        return None
    d = pd.to_datetime(entries["created_at"])
    span = (d.max() - d.min()).days + 1
    if span < 7:
        return None
    return float(entries["amount"].sum() / span * 30)


def next_buyer(paid_by: dict, people: list):
    """Whoever has contributed least in the current cycle should shop next."""
    if not people:
        return None
    return sorted(people, key=lambda n: (paid_by.get(n, 0), n))[0]


def to_ist(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.tz_localize("UTC").dt.tz_convert(IST).dt.strftime("%d %b, %I:%M %p")
