"""Database layer (SQLAlchemy). Works with local SQLite and hosted Postgres."""
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table,
                        create_engine, func, insert, select, update, delete)

meta = MetaData()
people = Table("people", meta,
    Column("id", Integer, primary_key=True),
    Column("name", String(60), unique=True, nullable=False))
cycles = Table("cycles", meta,
    Column("id", Integer, primary_key=True),
    Column("status", String(10), nullable=False),      # open | pending | closed
    Column("started_at", DateTime),
    Column("settled_at", DateTime),
    Column("total", Float), Column("share", Float))
entries = Table("entries", meta,
    Column("id", Integer, primary_key=True),
    Column("cycle_id", Integer, nullable=False, index=True),
    Column("person", String(60), nullable=False),
    Column("amount", Float, nullable=False),
    Column("item", String(120), nullable=False),
    Column("category", String(30), nullable=False),
    Column("flagged", Integer, default=0),
    Column("created_at", DateTime, nullable=False))
payments = Table("payments", meta,
    Column("id", Integer, primary_key=True),
    Column("cycle_id", Integer, nullable=False, index=True),
    Column("from_person", String(60), nullable=False),
    Column("to_person", String(60), nullable=False),
    Column("due", Float, nullable=False),
    Column("paid", Float, default=0),
    Column("mode", String(20)),
    Column("paid_at", DateTime))


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_engine(url=None):
    url = url or os.getenv("DATABASE_URL") or "sqlite:///flatsplit.db"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    kw = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kw["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kw)


class Store:
    def __init__(self, url=None):
        self.engine = get_engine(url)
        meta.create_all(self.engine)

    # ---- flatmates
    def people(self):
        with self.engine.connect() as c:
            return [r[0] for r in c.execute(select(people.c.name).order_by(people.c.id))]

    def set_people(self, names):
        with self.engine.begin() as c:
            if c.execute(select(func.count()).select_from(people)).scalar():
                return
            for n in dict.fromkeys(names):
                c.execute(insert(people).values(name=n))

    # ---- cycles and entries
    def open_cycle_id(self):
        with self.engine.begin() as c:
            r = c.execute(select(cycles.c.id).where(cycles.c.status == "open")).first()
            if r:
                return r[0]
            return c.execute(insert(cycles).values(status="open", started_at=utcnow())).inserted_primary_key[0]

    def add_entry(self, person, amount, item, category, flagged=False):
        cid = self.open_cycle_id()
        with self.engine.begin() as c:
            c.execute(insert(entries).values(cycle_id=cid, person=person, amount=amount, item=item,
                                             category=category, flagged=int(flagged), created_at=utcnow()))

    def delete_entry(self, entry_id):
        with self.engine.begin() as c:   # only entries of the open cycle can be deleted
            open_ids = select(cycles.c.id).where(cycles.c.status == "open")
            c.execute(delete(entries).where(entries.c.id == entry_id, entries.c.cycle_id.in_(open_ids)))

    def set_category(self, entry_id, category):
        with self.engine.begin() as c:
            c.execute(update(entries).where(entries.c.id == entry_id).values(category=category, flagged=0))

    def entries_df(self, cycle_id=None):
        q = select(entries).order_by(entries.c.created_at)
        if cycle_id is not None:
            q = q.where(entries.c.cycle_id == cycle_id)
        with self.engine.connect() as c:
            return pd.read_sql(q, c)

    def count_entries(self):
        with self.engine.connect() as c:
            return c.execute(select(func.count()).select_from(entries)).scalar()

    # ---- settlement and payments
    def settle(self, result):
        """Closes the open cycle, creates the payment list, starts a fresh cycle."""
        cid = self.open_cycle_id()
        with self.engine.begin() as c:
            if not c.execute(select(func.count()).select_from(entries).where(entries.c.cycle_id == cid)).scalar():
                return None
            status = "pending" if result["transfers"] else "closed"
            c.execute(update(cycles).where(cycles.c.id == cid).values(
                status=status, settled_at=utcnow(), total=result["total"], share=result["share"]))
            for t in result["transfers"]:
                c.execute(insert(payments).values(cycle_id=cid, from_person=t["from"], to_person=t["to"],
                                                  due=t["amount"], paid=0))
            c.execute(insert(cycles).values(status="open", started_at=utcnow()))
        return cid

    def cycles(self, status):
        with self.engine.connect() as c:
            return pd.read_sql(select(cycles).where(cycles.c.status == status).order_by(cycles.c.id.desc()), c)

    def payments_df(self, cycle_id):
        with self.engine.connect() as c:
            return pd.read_sql(select(payments).where(payments.c.cycle_id == cycle_id).order_by(payments.c.id), c)

    def record_payment(self, payment_id, amount, mode):
        with self.engine.begin() as c:
            p = c.execute(select(payments).where(payments.c.id == payment_id)).mappings().first()
            if not p:
                return
            new_paid = round(min(p["due"], (p["paid"] or 0) + amount), 2)
            c.execute(update(payments).where(payments.c.id == payment_id).values(
                paid=new_paid, mode=mode, paid_at=utcnow()))
            rows = c.execute(select(payments.c.due, payments.c.paid).where(payments.c.cycle_id == p["cycle_id"])).all()
            if all(d - (pd_ or 0) < 0.01 for d, pd_ in rows):
                c.execute(update(cycles).where(cycles.c.id == p["cycle_id"]).values(status="closed"))
