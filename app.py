"""FlatSplit - shared flat expense tracker with a learning categoriser.
Run locally:  streamlit run app.py"""
import os

import pandas as pd
import streamlit as st

import classifier
import logic
from db import Store
from report import make_pdf

st.set_page_config(page_title="FlatSplit", page_icon="🏠", layout="centered")


def cfg(key, default=None):
    v = os.getenv(key)
    if v:
        return v
    try:
        return st.secrets[key]
    except Exception:
        return default


@st.cache_resource
def get_store():
    return Store(cfg("DATABASE_URL"))


@st.cache_resource(max_entries=2)
def get_model(n_entries, _rows):
    return classifier.ItemClassifier().fit_seed_plus(_rows)


inr = lambda x: f"₹{x:,.2f}".replace(".00", "")
store = get_store()

# ---------- optional passcode
code = cfg("FLAT_PASSCODE")
if code and not st.session_state.get("ok"):
    st.title("🏠 FlatSplit")
    entered = st.text_input("Flat passcode", type="password")
    if entered and entered == code:
        st.session_state.ok = True
        st.rerun()
    elif entered:
        st.error("Wrong passcode")
    st.stop()

st.title("🏠 FlatSplit")

# ---------- first-time setup
names = store.people()
if not names:
    st.subheader("Who lives in the flat?")
    raw = st.text_input("Names separated by commas", placeholder="Priya, Riya, Sneha")
    if st.button("Save flatmates", type="primary"):
        n = [x.strip() for x in raw.split(",") if x.strip()]
        if len(n) < 2:
            st.error("Enter at least two names.")
        else:
            store.set_people(n)
            st.rerun()
    st.stop()

# ---------- who am I (kept in the page URL so each phone remembers it)
default = st.query_params.get("me")
idx = names.index(default) if default in names else 0
me = st.selectbox("I am", names, index=idx)
st.query_params["me"] = me

all_df = store.entries_df()
cid = store.open_cycle_id()
model = get_model(len(all_df), list(zip(all_df["item"], all_df["category"])) if len(all_df) else [])

t_add, t_settle, t_hist, t_ins = st.tabs(["➕ Add", "⚖️ Settle", "📜 History", "💡 Insights"])

# ================= ADD
with t_add:
    st.caption("Type amount and item, like **60 onions** or **40 milk**. Everyone's phone shows the same list "
               "(pull to refresh to see new entries).")
    with st.form("add", clear_on_submit=True):
        text = st.text_input("Amount and item", placeholder="60 onions")
        go = st.form_submit_button("Add expense", width="stretch", type="primary")
    if go:
        parsed = logic.parse_entry(text)
        if not parsed:
            st.error("Couldn't read that. Use the format: 60 onions")
        else:
            amount, item = parsed
            cat, conf = model.predict(item)
            flag = logic.is_anomaly(amount, cat, all_df)
            store.add_entry(me, amount, item, cat, flag)
            st.success(f"Added {inr(amount)} · {item} → {cat}")
            if flag:
                st.warning("That looks unusually high for this category. Delete it below if it was a typo.")

    open_df = store.entries_df(cid)
    st.subheader(f"Open expenses · {inr(open_df['amount'].sum() if len(open_df) else 0)}")
    if open_df.empty:
        st.info("Nothing logged yet in this cycle.")
    else:
        view = pd.DataFrame({
            "When": logic.to_ist(open_df["created_at"]), "Who": open_df["person"], "Item": open_df["item"],
            "Category": open_df["category"], "₹": open_df["amount"],
            "Check": open_df["flagged"].map({1: "⚠ high", 0: ""}),
        }).iloc[::-1]
        st.dataframe(view, hide_index=True, width="stretch")
        labels = {int(r.id): f"{r.person} · {r.item} · ₹{r.amount:g}" for r in open_df.itertuples()}
        with st.expander("Fix or delete an entry"):
            pick = st.selectbox("Entry", list(labels), format_func=labels.get, key="pick")
            c1, c2 = st.columns(2)
            newcat = c1.selectbox("Correct category", classifier.CATEGORIES, key="newcat")
            if c1.button("Save category"):
                store.set_category(pick, newcat)   # the model learns from this next run
                st.rerun()
            if c2.button("Delete entry"):
                store.delete_entry(pick)
                st.rerun()

# ================= SETTLE
with t_settle:
    for _, cy in store.cycles("pending").iterrows():
        st.subheader(f"Payments pending · settled {cy['settled_at']:%d %b}")
        st.caption(f"Total {inr(cy['total'])} · share {inr(cy['share'])} each")
        for p in store.payments_df(int(cy["id"])).itertuples():
            left = round(p.due - (p.paid or 0), 2)
            with st.container(border=True):
                st.markdown(f"**{p.from_person} → {p.to_person}** · due {inr(p.due)} · paid {inr(p.paid or 0)}")
                if left < 0.01:
                    st.success("Paid ✓")
                else:
                    with st.form(f"pay{p.id}"):
                        a, b = st.columns(2)
                        amt = a.number_input("Amount paid now", 0.0, float(left), float(left), 1.0)
                        mode = b.selectbox("Mode", ["UPI", "Cash", "Other"])
                        if st.form_submit_button("Record payment"):
                            if amt > 0:
                                store.record_payment(int(p.id), amt, mode)
                                st.rerun()
        st.divider()

    st.subheader("Settle now")
    open_df = store.entries_df(cid)
    if open_df.empty:
        st.info("No open expenses to settle.")
    else:
        paid_by = open_df.groupby("person")["amount"].sum().to_dict()
        res = logic.settle(paid_by, names)
        m1, m2 = st.columns(2)
        m1.metric("Total spent", inr(res["total"]))
        m2.metric("Fair share each", inr(res["share"]))
        st.dataframe(pd.DataFrame({"Person": res["names"], "Paid": [res["paid"][n] for n in res["names"]],
                                   "Balance": [round(res["paid"][n] - res["share"], 2) for n in res["names"]]}),
                     hide_index=True, width="stretch")
        if res["transfers"]:
            for t in res["transfers"]:
                st.markdown(f"**{t['from']}** pays **{t['to']}** {inr(t['amount'])}")
        else:
            st.success("Everyone is already even 🎉")
        agree = st.checkbox("Everyone agrees, settle now")
        if st.button("Settle & start payments", type="primary", disabled=not agree):
            store.settle(res)
            st.rerun()

# ================= HISTORY
with t_hist:
    closed = store.cycles("closed")
    if closed.empty:
        st.info("Closed settlements will appear here with every payment recorded.")
    for _, cy in closed.iterrows():
        ent, pay = store.entries_df(int(cy["id"])), store.payments_df(int(cy["id"]))
        title = f"{cy['settled_at']:%d %b %Y} · total {inr(cy['total'])} · {len(ent)} expenses"
        with st.expander(title):
            st.dataframe(pay.rename(columns={"from_person": "From", "to_person": "To", "due": "Due",
                                             "paid": "Paid", "mode": "Mode"})[["From", "To", "Due", "Paid", "Mode"]]
                         if len(pay) else pd.DataFrame({"": ["No payments were needed"]}),
                         hide_index=True, width="stretch")
            st.download_button("Download PDF report", make_pdf(cy, ent, pay),
                               file_name=f"settlement-{cy['settled_at']:%Y-%m-%d}.pdf",
                               mime="application/pdf", key=f"dl{cy['id']}")

# ================= INSIGHTS
with t_ins:
    if all_df.empty:
        st.info("Add a few expenses and insights will appear here.")
    else:
        open_df = store.entries_df(cid)
        pb = open_df.groupby("person")["amount"].sum().to_dict() if len(open_df) else {}
        st.metric("🛒 Who should buy next?", logic.next_buyer(pb, names))
        fc = logic.forecast_month(all_df)
        st.metric("📈 Projected spend next 30 days", inr(fc) if fc else "Needs 7+ days of data")
        st.subheader("Spending by category")
        st.bar_chart(all_df.groupby("category")["amount"].sum())
        st.subheader("Who spends the most")
        st.bar_chart(all_df.groupby("person")["amount"].sum())
        odd = all_df[all_df["flagged"] == 1]
        st.subheader("⚠ Possible mistakes")
        if len(odd):
            st.dataframe(odd[["person", "item", "category", "amount"]], hide_index=True)
        else:
            st.caption("No unusual entries detected.")
        st.caption(f"Categoriser trained on {len(classifier.ITEMS)} seed items + {len(all_df)} logged entries.")
