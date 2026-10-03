import { useMemo, useState } from "react";
import { imageUrl, Word } from "../api/client";

/** Highlighter colours, one per label. */
export const LABEL_COLORS: Record<string, string> = {
  VENDOR: "#a23b72",
  DATE: "#2c6fb0",
  INVOICE_NO: "#4a4ac0",
  ITEM_NAME: "#4f8a2f",
  ITEM_QTY: "#d08a12",
  ITEM_PRICE: "#8a4fb0",
  SUBTOTAL: "#13857f",
  TAX: "#94622c",
  TOTAL: "#d42a1c",
};

/** The page with tagged words highlighted. `focus` (the field being hovered/edited) dims every other label. */
export default function DocViewer({
  id,
  words,
  size,
  focus,
}: {
  id: string;
  words: Word[];
  size: { width: number; height: number };
  focus: string[];
}) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [zoom, setZoom] = useState(0); // 0 = fit width, otherwise a percentage
  const present = useMemo(() => Array.from(new Set(words.map((w) => w.label))).filter((l) => l in LABEL_COLORS), [words]);
  const toggle = (l: string) =>
    setHidden((h) => {
      const n = new Set(h);
      if (n.has(l)) n.delete(l);
      else n.add(l);
      return n;
    });
  const step = (d: number) => setZoom((z) => Math.min(300, Math.max(40, (z || 100) + d)));

  return (
    <>
      <div className="phead vtools">
        <b>Page</b>
        <span className="sp" />
        <button onClick={() => step(-20)} title="zoom out">−</button>
        <button onClick={() => setZoom(0)} title="fit to width">{zoom ? `${zoom}%` : "Fit"}</button>
        <button onClick={() => step(20)} title="zoom in">+</button>
      </div>
      <div className="chips">
        {present.map((l) => (
          <button key={l} className={`chip ${hidden.has(l) ? "off" : ""}`} onClick={() => toggle(l)} title="show / hide">
            <i style={{ background: LABEL_COLORS[l] }} />
            {l.replace("_", " ").toLowerCase()}
          </button>
        ))}
      </div>
      <div className="pbody">
        <div className="pagewrap">
          <div className="page" style={{ width: zoom ? `${zoom}%` : "100%", maxWidth: zoom ? "none" : 900 }}>
            <img src={imageUrl(id)} alt="document page" />
            <svg viewBox={`0 0 ${size.width} ${size.height}`} preserveAspectRatio="none">
              {words
                .filter((w) => w.label in LABEL_COLORS && !hidden.has(w.label))
                .map((w, i) => {
                  const on = focus.length === 0 || focus.includes(w.label);
                  return (
                    <rect
                      key={i}
                      x={w.box[0] - 1}
                      y={w.box[1]}
                      width={w.box[2] - w.box[0] + 2}
                      height={w.box[3] - w.box[1]}
                      fill={LABEL_COLORS[w.label]}
                      fillOpacity={on ? 0.35 : 0.06}
                    >
                      <title>{`${w.text} → ${w.label} (${(w.prob * 100).toFixed(0)}%)`}</title>
                    </rect>
                  );
                })}
            </svg>
          </div>
        </div>
      </div>
    </>
  );
}
