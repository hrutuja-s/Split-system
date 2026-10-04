import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import logic
from classifier import ItemClassifier
from db import Store


def test_parse():
    assert logic.parse_entry("60 onions") == (60.0, "onions")
    assert logic.parse_entry("₹40 milk") == (40.0, "milk")
    assert logic.parse_entry("milk 40") == (40.0, "milk")
    assert logic.parse_entry("2 kg onions 60") is None
    assert logic.parse_entry("hello") is None
    assert logic.parse_entry("0 milk") is None


def test_settlement_example():
    r = logic.settle({"A": 60, "B": 40, "C": 20}, ["A", "B", "C"])
    assert r["total"] == 120 and r["share"] == 40
    assert r["transfers"] == [{"from": "C", "to": "A", "amount": 20.0}]


def test_settlement_balances_to_zero_and_few_transfers():
    paid = {"A": 500, "B": 120, "C": 0, "D": 33.5}
    r = logic.settle(paid, list(paid))
    assert len(r["transfers"]) <= 3
    net = {n: paid[n] for n in paid}
    for t in r["transfers"]:
        net[t["from"]] += t["amount"]; net[t["to"]] -= t["amount"]
    assert all(abs(v - r["share"]) < 0.05 for v in net.values())


def test_anomaly():
    h = pd.DataFrame({"category": ["Vegetables"] * 6, "amount": [40, 60, 50, 45, 70, 55]})
    assert logic.is_anomaly(600, "Vegetables", h)
    assert not logic.is_anomaly(80, "Vegetables", h)


def test_classifier():
    m = ItemClassifier().fit_seed_plus([])
    assert m.predict("onions")[0] == "Vegetables"
    assert m.predict("milk")[0] == "Dairy"
    assert m.predict("sweets")[0] == "Snacks"
    assert m.predict("gas cylinder")[0] == "Household"


def test_full_flow(tmp_path):
    s = Store(f"sqlite:///{tmp_path}/t.db")
    s.set_people(["A", "B", "C"])
    for who, amt, item in [("A", 60, "onions"), ("B", 40, "milk"), ("C", 20, "sweets")]:
        s.add_entry(who, amt, item, "Other")
    cid = s.open_cycle_id()
    df = s.entries_df(cid)
    res = logic.settle(df.groupby("person")["amount"].sum().to_dict(), s.people())
    assert s.settle(res) == cid
    assert s.settle(res) is None            # new cycle is empty -> no double settle
    pay = s.payments_df(cid)
    assert len(pay) == 1 and len(s.cycles("pending")) == 1
    s.record_payment(int(pay.id[0]), 10, "UPI")
    assert len(s.cycles("pending")) == 1    # partial
    s.record_payment(int(pay.id[0]), 10, "Cash")
    assert len(s.cycles("closed")) == 1 and len(s.cycles("pending")) == 0
