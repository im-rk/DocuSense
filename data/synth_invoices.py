"""Synthetic labelled invoices/receipts (recommended training add-on).

We draw the pages ourselves, so every word, box and label is exact - free, perfect labels, and it adds the
INVOICE_NO class that CORD/SROIE lack. Mix ~30% synthetic into training; ALWAYS evaluate on real data only.
"""
import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from data.build_common import SplitBuilder
from models.tagger.labels import LABEL_ID as L

_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
          "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/cour.ttf", "C:/Windows/Fonts/times.ttf"]
_BOLD = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "C:/Windows/Fonts/arialbd.ttf"]

_A = ["Blue", "Green", "Prime", "Metro", "Star", "Royal", "Global", "Sun", "Delta", "Nova", "Apex", "Urban", "Silver",
      "Golden", "Swift", "Orbit", "Vertex", "Lotus", "Maple", "Cedar", "Indus", "Zenith"]
_B = ["Traders", "Solutions", "Foods", "Systems", "Enterprises", "Services", "Retail", "Supplies", "Logistics",
      "Consulting", "Mart", "Stores", "Technologies", "Hardware", "Bakery", "Cafe"]
_C = ["Pvt Ltd", "Ltd", "Sdn Bhd", "LLP", "Inc", "Co.", "& Sons", "Corp", "Bhd", "LLC"]
_ITEMS = ["Web Development", "QA Services", "Cloud Hosting", "Office Chair", "Printer Toner", "Consulting Hours",
          "A4 Paper Ream", "Laptop Repair", "Domain Renewal", "Data Entry", "Nasi Goreng", "Iced Latte", "Chicken Rice",
          "Steel Bolts", "Cable Ties", "Monthly Support", "Design Work", "Packaging", "Delivery Charge", "Software License"]
_CURR = ["", "$", "RM", "Rs.", "INR ", "USD "]
_ADDR = ["12 MG Road", "45 Jalan Ampang", "7 Park Street", "88 Market Lane", "3rd Floor, Tech Park", "21 Station Road"]


def _font(size, bold=False):
    paths = _BOLD if bold else random.sample(_FONTS, len(_FONTS))
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _money(v, cur, dec_style):
    s = f"{v:,.2f}"
    if dec_style == 1:
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    elif dec_style == 2:
        s = f"{v:.2f}"
    return f"{cur}{s}"


def _date(fmt):
    d, m, y = random.randint(1, 28), random.randint(1, 12), random.randint(2018, 2025)
    mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]
    return [f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{d} {mon} {y}", f"{mon} {d}, {y}", f"{d:02d}-{m:02d}-{y % 100:02d}"][fmt]


class _Canvas:
    def __init__(self, W, H):
        self.im, self.W, self.H = Image.new("L", (W, H), 255), W, H
        self.d, self.words = ImageDraw.Draw(self.im), []

    def text(self, s, x, y, font, label=0, right=False, ink=0):
        """Draw a phrase word by word (so each word gets its own exact box). Returns the end x."""
        gap = font.getlength(" ")
        widths = [font.getlength(w) for w in s.split()]
        total = sum(widths) + gap * max(len(widths) - 1, 0)
        if right:
            x = x - total
        for w, wd in zip(s.split(), widths):
            l, t, r, b = self.d.textbbox((x, y), w, font=font)
            self.d.text((x, y), w, font=font, fill=ink)
            self.words.append({"text": w, "box": [l, t, r, b], "label": label})
            x += wd + gap
        return x


def _degrade(im):
    if random.random() < 0.6:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.1)))
    a = np.asarray(im).astype(np.float32) * random.uniform(0.8, 1.05) + random.uniform(-15, 15)
    if random.random() < 0.6:
        a += np.random.normal(0, random.uniform(2, 8), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.5:                                        # JPEG artefacts
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=random.randint(40, 90)); im = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return im


def make_invoice(seed=None):
    """-> (PIL 'L' image, words[{text, box, label}]). Two layouts: A4 'table' invoice, narrow 'receipt'."""
    if seed is not None:
        random.seed(seed); np.random.seed(seed % (2 ** 32))
    receipt = random.random() < 0.35
    W, H = (random.randint(380, 480), random.randint(700, 900)) if receipt else (random.randint(760, 880), random.randint(950, 1150))
    cv = _Canvas(W, H)
    fs = random.randint(13, 18)
    f, fb, fbig = _font(fs), _font(fs + 2, True), _font(fs + random.randint(6, 12), True)
    cur, dec = random.choice(_CURR), random.choice([0, 0, 0, 1, 2])
    vendor = f"{random.choice(_A)} {random.choice(_B)} {random.choice(_C)}"
    m = 30
    y = m

    # header: vendor (labelled) + address
    align = random.choice(["left", "center", "right"]) if not receipt else "center"
    vw = sum(fbig.getlength(w) + fbig.getlength(" ") for w in vendor.split())
    x = {"left": m, "center": (W - vw) / 2, "right": W - m - vw}[align]
    cv.text(vendor, x, y, fbig, L["VENDOR"]); y += fbig.size + 8
    cv.text(f"{random.choice(_ADDR)}, Ph {random.randint(2000000, 9999999)}", m, y, f); y += fs + 14
    if not receipt:
        cv.text(random.choice(["INVOICE", "TAX INVOICE", "BILL"]), W - m, y, fbig, right=True); y += fbig.size + 12

    # invoice number + date (key text = OTHER, value = labelled)
    inv = random.choice(["INV-", "INV", "#", "IN/", "BL-"]) + f"{random.randint(2018, 2025)}-{random.randint(1, 9999):04d}"
    kx = cv.text(random.choice(["Invoice No:", "Inv #:", "Bill No.", "Invoice Number:", "Receipt No:"]), m, y, fb)
    cv.text(inv, kx + 8, y, f, L["INVOICE_NO"])
    y += fs + 8
    kx = cv.text(random.choice(["Date:", "Invoice Date:", "Dated:"]), m, y, fb)
    cv.text(_date(random.randint(0, 4)), kx + 8, y, f, L["DATE"]); y += fs + 22

    # items table
    n = random.randint(2, 7 if not receipt else 9)
    cx_desc, cx_qty, cx_rate, cx_amt = m, W * 0.52, W * 0.68, W - m
    if not receipt:
        for txt, xx, rt in (("Description", cx_desc, False), ("Qty", cx_qty, False), ("Rate", cx_rate, False), ("Amount", cx_amt, True)):
            cv.text(txt, xx, y, fb, right=rt)
        y += fs + 12
    sub = 0.0
    for name in random.sample(_ITEMS, n):
        qty, rate = random.randint(1, 12), round(random.uniform(2, 900), 2)
        amt = round(qty * rate, 2); sub += amt
        cv.text(name, cx_desc, y, f, L["ITEM_NAME"])
        if not receipt:
            cv.text(str(qty), cx_qty, y, f, L["ITEM_QTY"])
            cv.text(_money(rate, cur, dec), cx_rate, y, f)                        # unit rate: OTHER
        else:
            cv.text(f"{qty}x", cx_qty - 20, y, f, L["ITEM_QTY"])
        cv.text(_money(amt, cur, dec), cx_amt, y, f, L["ITEM_PRICE"], right=True)
        y += fs + 9
    y += 12

    # totals block, right-aligned
    rate_t = random.choice([5, 6, 8, 10, 12, 18])
    tax = round(sub * rate_t / 100, 2)
    tot = round(sub + tax, 2)
    for key, val, lab in ((random.choice(["Subtotal", "Sub Total", "Net Amount"]), sub, L["SUBTOTAL"]),
                          (random.choice([f"GST {rate_t}%", f"Tax ({rate_t}%)", f"VAT {rate_t}%", f"SST {rate_t}%"]), tax, L["TAX"]),
                          (random.choice(["Total", "Grand Total", "TOTAL DUE", "Amount Payable"]), tot, L["TOTAL"])):
        bold = lab == L["TOTAL"]
        cv.text(key, cx_amt - (fb.getlength("0" * 12)) - 60, y, fb if bold else f)
        cv.text(_money(val, cur, dec), cx_amt, y, fb if bold else f, lab, right=True)
        y += fs + 9
    y += 18
    for line in random.sample(["Thank you for your business", "Payment due in 30 days", "Goods once sold cannot be returned",
                               f"Bank A/C {random.randint(10**9, 10**10 - 1)}", "This is a computer generated invoice"], 2):
        if y + fs < H - 5:
            cv.text(line, m, y, f); y += fs + 6
    return _degrade(cv.im), cv.words


def build_synth(out_dir, n_train=2000, n_val=200, audit=10, seed=1234):
    stats = []
    for name, n, off in (("synth_train", n_train, 0), ("synth_val", n_val, 10_000_000)):
        sb = SplitBuilder(out_dir, name, audit=audit, keep_receipt=False)
        for i in range(n):
            im, words = make_invoice(seed + off + i)
            sb.add(f"{name}_{i:05d}", "synth", name.split("_")[1], np.asarray(im), words, pil_img=im.convert("RGB"))
        stats.append(sb.save()); print(stats[-1])
    return stats
