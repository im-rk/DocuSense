import { useEffect, useState } from "react";
import { LineItem, Result, saveFields } from "../api/client";

/** field key, label, and the model label it comes from (used to light up the page while a field is focused). */
export const FIELDS: [string, string, string][] = [
  ["vendor", "Vendor", "VENDOR"],
  ["date", "Date", "DATE"],
  ["invoice_number", "Invoice no.", "INVOICE_NO"],
  ["subtotal", "Subtotal", "SUBTOTAL"],
  ["tax", "Tax", "TAX"],
  ["total", "Total", "TOTAL"],
];
const ITEM_LABELS = ["ITEM_NAME", "ITEM_QTY", "ITEM_PRICE"];

const show = (v: unknown) => (v === undefined || v === null ? "" : String(v));

/** Editable fields + line items. Saving stores an audited correction and re-runs validation. */
export default function FieldTable({
  id,
  result,
  onSaved,
  onFocus,
  focus,
  saveRef,
}: {
  id: string;
  result: Result;
  onSaved: () => void;
  onFocus: (labels: string[]) => void;
  focus: string[];
  saveRef: React.MutableRefObject<(() => void) | null>;
}) {
  const initial = () => Object.fromEntries(FIELDS.map(([k]) => [k, show(result.fields[k])]));
  const [vals, setVals] = useState<Record<string, string>>(initial);
  const [items, setItems] = useState<LineItem[]>(result.line_items);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setVals(initial());
    setItems(result.line_items);
    setErr("");
  }, [result]); // eslint-disable-line react-hooks/exhaustive-deps

  const changed = FIELDS.filter(([k]) => vals[k] !== show(result.fields[k]));
  const itemsChanged = JSON.stringify(items) !== JSON.stringify(result.line_items);
  const dirty = changed.length > 0 || itemsChanged;

  async function save() {
    if (!dirty || busy) return;
    setBusy(true);
    setErr("");
    try {
      await saveFields(
        id,
        Object.fromEntries(changed.map(([k]) => [k, vals[k].trim() === "" ? null : vals[k]])),
        itemsChanged ? items : undefined,
      );
      onSaved();
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (e) {
      setErr((e as Error).message);
    }
    setBusy(false);
  }
  saveRef.current = () => void save();

  const setItem = (i: number, patch: Partial<LineItem>) => setItems(items.map((it, j) => (j === i ? { ...it, ...patch } : it)));
  const num = (s: string) => (s.trim() === "" ? null : Number(s));

  return (
    <>
      <div className="pbody" onMouseLeave={() => onFocus([])}>
        <div className="sec">Fields</div>
        {FIELDS.map(([k, label, lab]) => {
          const c = result.field_confidences[k];
          return (
            <div className={`fld ${focus[0] === lab ? "hot" : ""}`} key={k} onMouseEnter={() => onFocus([lab])}>
              <label htmlFor={`f-${k}`}>{label}</label>
              <input
                id={`f-${k}`}
                className={`ed ${vals[k] !== show(result.fields[k]) ? "changed" : ""} ${vals[k] === "" ? "na" : ""}`}
                value={vals[k] ?? ""}
                placeholder="not found"
                onFocus={() => onFocus([lab])}
                onChange={(e) => setVals({ ...vals, [k]: e.target.value })}
              />
              <span className={`conf ${c !== undefined && c < 0.7 ? "lo" : c !== undefined && c < 0.9 ? "mid" : ""}`} title="model confidence, %">
                {c === undefined ? "" : (c * 100).toFixed(0)}
              </span>
            </div>
          );
        })}

        <div className="sec" style={{ marginTop: 10 }} onMouseEnter={() => onFocus(ITEM_LABELS)}>
          <span>Line items</span>
          <span className="dim num">{items.length}</span>
        </div>
        {items.length > 0 && (
          <table className="items" onMouseEnter={() => onFocus(ITEM_LABELS)}>
            <thead>
              <tr>
                <th>Item</th>
                <th>Qty</th>
                <th>Price</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {items.map((it, i) => (
                <tr key={i}>
                  <td>
                    <input className="ed" value={it.name} onChange={(e) => setItem(i, { name: e.target.value })} />
                  </td>
                  <td className="q">
                    <input className="ed num" value={it.qty ?? ""} onChange={(e) => setItem(i, { qty: num(e.target.value) })} />
                  </td>
                  <td className="p">
                    <input className="ed num" value={it.price ?? ""} onChange={(e) => setItem(i, { price: num(e.target.value) })} />
                  </td>
                  <td className="x">
                    <button title="remove row" onClick={() => setItems(items.filter((_, j) => j !== i))}>
                      ×
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <button className="link" onClick={() => setItems([...items, { name: "", qty: null, price: null }])}>
          Add row
        </button>
        <Checks result={result} />
      </div>
      <div className="sticky">
        <button className="btn dark" disabled={!dirty || busy} onClick={() => void save()} title="Ctrl+S">
          {busy ? "Saving…" : "Save changes"}
        </button>
        {saved && <span className="saved">Saved</span>}
        {err && <span style={{ color: "var(--bad)" }}>{err}</span>}
        <span className="sp" />
      </div>
    </>
  );
}

function Checks({ result }: { result: Result }) {
  const v = result.validation;
  return (
    <>
      <div className="sec" style={{ marginTop: 10 }}>Checks</div>
      {Object.entries(v.checks).map(([k, s]) => (
        <div className="check" key={k}>
          <b className={s}>{s}</b>
          <span>{k.replace(/_/g, " ")}</span>
        </div>
      ))}
      {v.errors.map((e) => (
        <div key={e} className="msg bad">{e}</div>
      ))}
      {v.warnings.map((w) => (
        <div key={w} className="msg">{w}</div>
      ))}
      {v.errors.length + v.warnings.length === 0 && <div className="msg good">All checks passed.</div>}
    </>
  );
}
