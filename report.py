"""PDF settlement report (fpdf2). Core fonts have no rupee glyph, so 'Rs' is used."""
from fpdf import FPDF

from logic import to_ist


def make_pdf(cycle, ent, pay) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "FlatSplit - Settlement Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    start = ent["created_at"].min().strftime("%d %b %Y") if len(ent) else "-"
    end = cycle["settled_at"].strftime("%d %b %Y") if cycle["settled_at"] is not None else "-"
    pdf.cell(0, 7, f"Period: {start} to {end}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Total spent: Rs {cycle['total']:.2f}    Share per person: Rs {cycle['share']:.2f}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Expenses", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    shown = to_ist(ent["created_at"]) if len(ent) else []
    for (_, r), when in zip(ent.iterrows(), shown):
        pdf.cell(0, 6, f"{when}   {r['person']}   {r['item']} ({r['category']})   Rs {r['amount']:.2f}",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Payments", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    if not len(pay):
        pdf.cell(0, 6, "Everyone was already even - no payments needed.", new_x="LMARGIN", new_y="NEXT")
    for _, r in pay.iterrows():
        mode = f" via {r['mode']}" if r["mode"] else ""
        pdf.cell(0, 6, f"{r['from_person']} -> {r['to_person']}: due Rs {r['due']:.2f}, paid Rs {r['paid']:.2f}{mode}",
                 new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
