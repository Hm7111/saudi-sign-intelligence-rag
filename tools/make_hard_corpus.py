"""Build a HARDER synthetic MC3 corpus + 20 graded questions (the real graded set is "larger and harder").
Traps: superseded revisions flagged only in their header text, near-identical products, multi-sheet xlsx with near-miss
rows, a 3-file chain, a scanned (image-only) PDF page, a rotated label photo, a pinout picture, code defaults vs comments,
long logs that push the corpus past the full-context size, encrypted / unknown / empty cases, two unanswerables.
usage: python make_hard_corpus.py <out_dir>      -> <out_dir>/corpus/... and <out_dir>/hard-questions.json
"""
import os, sys, json, random, csv
import pymupdf, docx, openpyxl
from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1]; C = os.path.join(out, "corpus"); random.seed(7)
def P(*a):
    p = os.path.join(C, *a); os.makedirs(os.path.dirname(p), exist_ok=True); return p
def font(sz):
    for f in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()

def pdf(path, pages, encrypt=False):
    d = pymupdf.open()
    for text in pages:
        pg = d.new_page()
        pg.insert_textbox(pymupdf.Rect(50, 50, 560, 800), text, fontsize=10, fontname="helv")
    if encrypt:
        d.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="s3cret", owner_pw="own3r")
    else:
        d.save(path)

# ---------------- specs ----------------
pdf(P("specs", "hx300_datasheet_r1.pdf"), ["""Halcyon Dynamics HX-300 "Corvid" Inference Card
Datasheet - Revision 1. Superseded by revision 3; retained for audit only.
Electrical and thermal
Maximum junction temperature ........ 105 C
Typical board power ................. 410 W
Memory capacity ..................... 80 GiB"""])
pdf(P("specs", "hx300_datasheet_r2_OBSOLETE.pdf"), ["""Halcyon Dynamics HX-300 "Corvid" Inference Card
Datasheet - Revision 2 (obsolete)
Maximum junction temperature ........ 100 C
Typical board power ................. 440 W"""])
pdf(P("specs", "hx300_datasheet_r3.pdf"), ["""Halcyon Dynamics HX-300 "Corvid" Inference Card
Datasheet - Revision 3. Supersedes revisions 1 and 2.
1. Electrical and thermal
Maximum junction temperature ........ 97 C
Typical board power ................. 450 W
Idle power .......................... 38 W
Auxiliary connector ................. 12V-2x6
""", """2. Memory and interfaces
Memory capacity ..................... 96 GiB HBM3e
Peak memory bandwidth ............... 5.3 TB/s
Host interface ...................... PCIe Gen5 x16
3. Revision history
r3  Junction limit lowered to 97 C after qualification. Revisions 1 and 2 are withdrawn.
r2  Board power update.
r1  Preliminary."""])
pdf(P("specs", "hx500_datasheet.pdf"), ["""Halcyon Dynamics HX-500 "Raptor" Training Card - Datasheet, revision 1
Maximum junction temperature ........ 101 C
Typical board power ................. 700 W
Memory capacity ..................... 192 GiB HBM3e"""])
pdf(P("specs", "hx200_product_brief.pdf"), ["""HX-200 "Wren" Edge Accelerator - Product Brief
Form factor: low-profile, single slot. Passive cooling for fanless gateways.
Typical board power: 75 W. Memory capacity: 24 GiB.""",
"""HX-200 environmental
Operating temperature ............... 0 to 55 C
Acoustic noise at full load ......... 41 dBA (with optional fan kit FK-2)
Shock ............................... 30 G, 11 ms"""])
# scanned page: text exists only as pixels
img = Image.new("RGB", (1240, 800), "white"); dr = ImageDraw.Draw(img)
dr.text((60, 60), "Thermal qualification report - HX-300 chassis TQ-7 (scanned copy)", fill="black", font=font(34))
dr.text((60, 180), "Maximum inlet air temperature: 35 C", fill="black", font=font(40))
dr.text((60, 260), "Minimum airflow: 180 CFM at 30 C inlet", fill="black", font=font(34))
dr.text((60, 340), "Signed: Thermal lab, 2026-08-21", fill="black", font=font(28))
img = img.rotate(1.2, expand=True, fillcolor="white"); bpath = os.path.join(out, "_scan.png"); img.save(bpath)
d = pymupdf.open(); pg = d.new_page(); pg.insert_image(pymupdf.Rect(30, 30, 580, 420), filename=bpath); d.save(P("specs", "thermal_qualification_scan.pdf"))
# pinout picture
img = Image.new("RGB", (1100, 520), "white"); dr = ImageDraw.Draw(img)
dr.text((40, 30), "HX-300 fan header J7 - pin assignment", fill="black", font=font(36))
sig = ["GND", "+12V", "FAN_TACH1", "FAN_TACH2", "FAN_PWM", "GND"]
for i, s in enumerate(sig):
    x = 40 + i * 175
    dr.rectangle([x, 140, x + 140, 220], outline="black", width=3)
    dr.text((x + 30, 160), f"J7-{i + 1}", fill="black", font=font(32))
    dr.line([x + 70, 220, x + 70, 260], fill="black", width=2)
    dr.text((x + 5, 270), s, fill="black", font=font(22))
dr.text((40, 380), "FAN_TACH2 reports the rear rotor. Pins J7-1 and J7-6 are ground.", fill="black", font=font(26))
img.save(P("specs", "fan_header_j7.png"))
# rotated label photo
img = Image.new("RGB", (900, 520), (225, 225, 220)); dr = ImageDraw.Draw(img)
dr.rectangle([60, 60, 840, 460], fill="white", outline="black", width=4)
for y, t in [(100, "Halcyon Dynamics"), (180, "MODEL  HX-300"), (250, "LOT  2026-W37-B"), (320, "SN  HD3-552019-04"), (390, "ASSEMBLED IN VIETNAM")]:
    dr.text((100, y), t, fill="black", font=font(40 if y < 120 else 36))
img = img.rotate(-4, expand=True, fillcolor=(225, 225, 220)); img.save(P("support", "rack_label_photo.jpg"), quality=88)

# ---------------- planning (docx) ----------------
dd = docx.Document(); dd.add_heading("Halcyon Dynamics - Product Roadmap FY28 (internal)", 1)
dd.add_paragraph("Dates are targets. Milestones below supersede the FY27 plan.")
t = dd.add_table(rows=1, cols=3); t.rows[0].cells[0].text, t.rows[0].cells[1].text, t.rows[0].cells[2].text = "Product", "Milestone", "Target quarter"
for r in [("HX-500", "Customer sampling", "Q4 FY27"), ("HX-500", "General availability", "Q2 FY28"), ("HX-300", "End of life notice", "Q4 FY29"), ("HX-200", "Fan kit FK-3", "Q1 FY28")]:
    c = t.add_row().cells; c[0].text, c[1].text, c[2].text = r
dd.save(P("planning", "roadmap_fy28.docx"))
dd = docx.Document(); dd.add_heading("Staff meeting notes - August 2026", 1)
dd.add_paragraph("HX-500 general availability was discussed at length; finance asked for an earlier date, nothing was decided in this meeting. "
                 "The team agreed to keep the roadmap document as the single source of truth for dates.")
dd.add_paragraph("Action: S. Rahman to circulate the revised fan kit plan.")
dd.save(P("planning", "meeting_notes_2026-08.docx"))
dd = docx.Document(); dd.add_heading("Commercial policy", 1)
dd.add_paragraph("Discounts above the standard threshold need VP Sales approval. The standard threshold is 18 percent off list price.")
dd.save(P("planning", "pricing_policy.docx"))

# ---------------- finance (xlsx, csv) ----------------
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "HX-300 BOM"
ws.append(["part_number", "description", "supplier", "unit_cost_usd", "lead_time_days"])
for r in [("HD-VC-3001", "Vapor chamber", "Kestrel Thermal", 96.0, 35), ("HD-FAN-3002", "Blower fan, dual", "Aerin Motors", 22.5, 14),
          ("HD-HS-3003", "Heatsink, fin stack", "Kestrel Thermal", 41.0, 21)]:
    ws.append(r)
ws = wb.create_sheet("HX-500 BOM"); ws.append(["part_number", "description", "supplier", "unit_cost_usd", "lead_time_days"])
for r in [("HD-VC-5001", "Vapor chamber", "Northgate Cooling", 158.0, 42), ("HD-FAN-5002", "Blower fan, triple", "Aerin Motors", 31.0, 18),
          ("HD-HS-5003", "Heatsink, fin stack", "Northgate Cooling", 77.0, 28)]:
    ws.append(r)
ws = wb.create_sheet("Notes"); ws.append(["Costs are FY27 contract prices. Lead times are supplier-quoted."])
wb.save(P("finance", "bom_costs.xlsx"))
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Summary"; ws.append(["month", "total_returns"]); ws.append(["2026-07", 40]); ws.append(["2026-08", 33])
ws = wb.create_sheet("By product"); ws.append(["month", "product", "returns", "top_cause"])
for r in [("2026-07", "HX-300", 21, "fan noise"), ("2026-07", "HX-500", 19, "PSU"), ("2026-08", "HX-300", 17, "fan noise"), ("2026-08", "HX-500", 16, "thermal")]:
    ws.append(r)
wb.save(P("support", "rma_returns.xlsx"))
with open(P("finance", "budget_fy27.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cost_center", "owner", "budget_usd"])
    for r in [("CC-410", "Hardware", 1200000), ("CC-420", "Firmware", 860000), ("CC-430", "Support", 410000)]:
        w.writerow(r)

# ---------------- support ----------------
with open(P("support", "tickets.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ticket", "component", "severity", "summary", "status", "fixed_in", "workaround_ref"])
    w.writerows([("HAL-2291", "thermal", "S2", "Rear rotor tach reads zero intermittently under burst load", "closed", "7.2.1", "KB-117"),
                 ("HAL-2240", "pcie", "S3", "Link retrains after warm reset", "closed", "7.1.4", ""),
                 ("HAL-2305", "scheduler", "S3", "Queue depth metric off by one", "open", "", ""),
                 ("HAL-2188", "thermal", "S4", "Fan curve too aggressive at idle", "closed", "6.9.0", "KB-102")])
with open(P("support", "kb_articles.txt"), "w") as f:
    f.write("Halcyon knowledge base (support)\n\n"
            "KB-102  Fan curve at idle. Workaround: set idle_fan_pct=25 until the fix is installed.\n"
            "KB-117  Tach dropouts on the rear rotor. Workaround: raise fan_floor_pct to 35 so the rotor never stalls.\n"
            "KB-121  PCIe retrain after warm reset. Workaround: cold boot.\n")

# ---------------- logs (long: pushes the corpus past full-context size) ----------------
def noise_lines(node, day, n):
    out_l = []
    for i in range(n):
        h, m, s = 1 + i // 60 % 22, i % 60, (i * 7) % 60
        out_l.append(f"2026-09-{day}T{h:02d}:{m:02d}:{s:02d}Z {node} nimbus[1871]: INFO  scheduler: batch {50000 + i} accepted, {random.randint(8, 96)} requests, p99 {random.randint(40, 90)} ms")
    return out_l
L = noise_lines("node-12", "14", 700)
L.insert(512, "2026-09-14T09:31:07Z node-12 nimbus[1871]: WARN  W4410: rear rotor tach read 0 rpm for 3 s, fan floor applied")
L.insert(513, "2026-09-14T09:31:07Z node-12 nimbus[1871]: INFO  incident linked to HAL-2291")
open(P("logs", "node-12_2026-09-14.log"), "w").write("\n".join(L) + "\n")
L = noise_lines("node-03", "10", 600)
L.insert(300, "2026-09-10T05:02:44Z node-03 nimbus[1690]: ERROR E5120: PCIe link retrained after warm reset")
L.insert(301, "2026-09-10T05:02:44Z node-03 nimbus[1690]: INFO  incident linked to HAL-2240")
open(P("logs", "node-03_2026-09-10.log"), "w").write("\n".join(L) + "\n")

# ---------------- engineering ----------------
open(P("engineering", "scheduler_config.py"), "w").write('''"""Nimbus scheduler configuration."""
# Back-off between retries, seconds. Was 30 before the 7.0 release; shortened after load tests.
RETRY_BACKOFF_S = 12
MAX_QUEUE_DEPTH = 4096
PREEMPT_ENABLED = True
''')
open(P("engineering", "telemetry_agent.py"), "w").write('''"""Telemetry agent: samples sensors and ships them to the collector."""
import time


def run(collector, sample_interval_ms=250, batch=64):
    """Sample every sample_interval_ms (it used to be 500 ms in agent 1.x)."""
    while True:
        collector.ship([read_sensors() for _ in range(batch)])
        time.sleep(sample_interval_ms / 1000)
''')
open(P("engineering", "release_notes_nimbus.txt"), "w").write("Nimbus firmware release notes\n\n" + "\n".join(
    f"7.{a}.{b}  Maintenance release: stability and telemetry improvements." for a in range(0, 3) for b in range(0, 5)) +
    "\n\nThe ticket tracker is authoritative for which release closed a ticket.\n")
open(P("engineering", "ci_build.yaml"), "w").write("pipeline:\n  name: nimbus-firmware\n  timeout_minutes: 45\n  runners: 6\n")

# ---------------- traps ----------------
pdf(P("vendor", "supplier_nda_ENCRYPTED.pdf"), ["HX-300 unit price at 1,000 units: 2,380 USD"], encrypt=True)
open(P("vendor", "firmware_blob.bin"), "wb").write(bytes(range(256)) * 64)
open(P("vendor", "audit_restricted.txt"), "w").write("Restricted. HX-300 MTBF: 210000 hours.\n")   # made unreadable by the runner (chmod 000)
os.makedirs(P("archive", "x").rsplit(os.sep, 1)[0], exist_ok=True)
os.makedirs(os.path.join(C, "tmp", "old"), exist_ok=True)

Q = [
 ("What is the maximum junction temperature of the HX-300?", "97", ["specs/hx300_datasheet_r3.pdf"]),
 ("What is the idle power of the HX-300 in watts?", "38", ["specs/hx300_datasheet_r3.pdf"]),
 ("What is the peak memory bandwidth of the HX-300 in TB/s?", "5.3", ["specs/hx300_datasheet_r3.pdf"]),
 ("What is the maximum junction temperature of the HX-500?", "101", ["specs/hx500_datasheet.pdf"]),
 ("What is the acoustic noise of the HX-200 at full load, in dBA?", "41", ["specs/hx200_product_brief.pdf"]),
 ("What is the maximum inlet air temperature in the thermal qualification report?", "35", ["specs/thermal_qualification_scan.pdf"]),
 ("Which pin of fan header J7 carries FAN_TACH2?", "J7-4", ["specs/fan_header_j7.png"]),
 ("What lot code is printed on the rack label?", "2026-W37-B", ["support/rack_label_photo.jpg"]),
 ("In which quarter does the HX-500 reach general availability?", "Q2 FY28", ["planning/roadmap_fy28.docx"]),
 ("What is the lead time in days for the HX-500 vapor chamber?", "42", ["finance/bom_costs.xlsx"]),
 ("Which supplier provides the HX-300 vapor chamber?", "Kestrel Thermal", ["finance/bom_costs.xlsx"]),
 ("How many HX-300 units were returned in August 2026?", "17", ["support/rma_returns.xlsx"]),
 ("Which firmware release fixed ticket HAL-2240?", "7.1.4", ["support/tickets.csv"]),
 ("Node 12 logged a warning on 2026-09-14. Which firmware release fixed the underlying defect?", "7.2.1",
  ["logs/node-12_2026-09-14.log", "support/tickets.csv"]),
 ("What fan floor percentage does the workaround recommend for the problem behind warning W4410?", "35",
  ["logs/node-12_2026-09-14.log", "support/tickets.csv", "support/kb_articles.txt"]),
 ("What is the retry backoff, in seconds, in the scheduler configuration?", "12", ["engineering/scheduler_config.py"]),
 ("What is the default sample interval of the telemetry agent, in milliseconds?", "250", ["engineering/telemetry_agent.py"]),
 ("What is the CI pipeline timeout in minutes?", "45", ["engineering/ci_build.yaml"]),
 ("What is the unit price of the HX-300 at 1,000 units?", "", []),
 ("What is the MTBF of the HX-300 in hours?", "", []),
]
json.dump({"queries": [{"n": i + 1, "query": q, "expected_answer": a, "answer_aliases": [], "expected_citations": c}
                       for i, (q, a, c) in enumerate(Q)]}, open(os.path.join(out, "hard-questions.json"), "w"), indent=1)
os.remove(bpath)
tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(C) for f in fs)
print("corpus files:", sum(len(fs) for _, _, fs in os.walk(C)), "bytes:", tot, "questions:", len(Q))
