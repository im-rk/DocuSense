import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { deleteDoc, DocDetail, DocSummary, exportUrl, getDoc, listDocs } from "../api/client";
import DocViewer from "../components/DocViewer";
import FieldTable from "../components/FieldTable";

const BUSY = ["PENDING", "PROCESSING"];
const label = (s: string) => s.replace("_", " ").toLowerCase().replace(/^\w/, (c) => c.toUpperCase());

/** List | page | fields. Keyboard: ↑/↓ or J/K move through documents, Ctrl+S saves. */
export default function Workspace({
  selectedId,
  onSelect,
  refreshKey,
  toast,
}: {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  refreshKey: number;
  toast: (msg: string, bad?: boolean) => void;
}) {
  const [docs, setDocs] = useState<DocSummary[] | null>(null);
  const [q, setQ] = useState("");
  const [detail, setDetail] = useState<DocDetail | null>(null);
  const [focus, setFocus] = useState<string[]>([]);
  const saveRef = useRef<(() => void) | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDocs(await listDocs());
    } catch (e) {
      toast(`Cannot reach the API: ${(e as Error).message}`, true);
      setDocs((d) => d ?? []);
    }
  }, [toast]);

  const loadDetail = useCallback(async () => {
    if (!selectedId) return setDetail(null);
    try {
      setDetail(await getDoc(selectedId));
    } catch {
      setDetail(null);
      onSelect(null);
    }
  }, [selectedId, onSelect]);

  useEffect(() => { void refresh(); }, [refresh, refreshKey]);
  useEffect(() => { setDetail(null); void loadDetail(); }, [selectedId, refreshKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // poll while anything is still being processed
  const busy = docs?.some((d) => BUSY.includes(d.status)) || (detail && BUSY.includes(detail.document.status));
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => { void refresh(); void loadDetail(); }, 1500);
    return () => clearInterval(t);
  }, [busy, refresh, loadDetail]);

  const shown = useMemo(() => {
    const s = q.trim().toLowerCase();
    return (docs ?? []).filter((d) => !s || [d.name, d.vendor, d.document_type].some((x) => x?.toLowerCase().includes(s)));
  }, [docs, q]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        saveRef.current?.();
        return;
      }
      const t = e.target as HTMLElement;
      if (t.tagName === "INPUT" || !shown.length) return;
      const i = shown.findIndex((d) => d.id === selectedId);
      if (e.key === "ArrowDown" || e.key === "j") onSelect(shown[Math.min(shown.length - 1, i + 1)].id);
      else if (e.key === "ArrowUp" || e.key === "k") onSelect(shown[Math.max(0, i - 1)].id);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [shown, selectedId, onSelect]);

  const remove = async (d: DocSummary) => {
    if (!window.confirm(`Delete "${d.name}"?`)) return;
    await deleteDoc(d.id).catch((e: Error) => toast(e.message, true));
    if (d.id === selectedId) onSelect(null);
    void refresh();
  };

  const r = detail?.result;
  const d = detail?.document;

  return (
    <div className="work">
      {/* ---- list ---- */}
      <section className="pane">
        <div className="phead">
          <b>Documents</b>
          <span className="dim num">{docs?.length ?? ""}</span>
        </div>
        <div className="search">
          <input placeholder="Search name, vendor, type" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <div className="pbody">
          {docs && docs.length === 0 && (
            <div className="empty">
              <b>No documents yet</b>Use Upload, or drag files anywhere onto this window.
            </div>
          )}
          {docs && docs.length > 0 && shown.length === 0 && <div className="empty">Nothing matches “{q}”.</div>}
          {shown.map((x) => (
            <div key={x.id} className={`row ${x.id === selectedId ? "sel" : ""}`} onClick={() => onSelect(x.id)}>
              <i className={`dot ${x.status}`} title={label(x.status)} />
              <div style={{ minWidth: 0 }}>
                <div className="t1">{x.vendor || x.name}</div>
                <div className="t2">{x.vendor ? x.name : label(x.status)}{x.document_type ? ` · ${x.document_type.toLowerCase()}` : ""}</div>
              </div>
              <div className="amt">{x.total != null ? x.total.toFixed(2) : ""}</div>
              <button className="del" title="Delete" onClick={(e) => { e.stopPropagation(); void remove(x); }}>×</button>
            </div>
          ))}
        </div>
      </section>

      {/* ---- page ---- */}
      <section className="pane mid">
        {!d && <div className="placeholder"><div><b>{docs?.length ? "Select a document" : "Nothing to show"}</b>The page appears here with every recognised word highlighted.</div></div>}
        {d && BUSY.includes(d.status) && <div className="working">Reading the page…</div>}
        {d?.status === "FAILED" && <div className="msg bad" style={{ margin: 16 }}>Processing failed: {d.error}</div>}
        {d && r && <DocViewer id={d.id} words={r.words} size={r.image_size} focus={focus} />}
      </section>

      {/* ---- fields ---- */}
      <section className="pane">
        <div className="phead">
          <b>{d ? d.name : "Details"}</b>
          <span className="sp" />
          {d && <span className="dim"><i className={`dot ${d.status}`} />{label(d.status)}</span>}
        </div>
        {d && r ? (
          <>
            <div className="phead" style={{ background: "transparent", height: 28 }}>
              <span className="dim">
                {r.document_type?.toLowerCase()} · {((r.type_confidence ?? 0) * 100).toFixed(0)}% · {r.words.length} words · {r.inference_ms} ms
              </span>
              <span className="sp" />
              {(["json", "csv", "xlsx"] as const).map((f) => (
                <a key={f} className="btn plain" style={{ padding: "1px 7px", textDecoration: "none" }} href={exportUrl(d.id, f)}>{f}</a>
              ))}
            </div>
            <FieldTable id={d.id} result={r} onSaved={() => { void loadDetail(); void refresh(); }} onFocus={setFocus} focus={focus} saveRef={saveRef} />
          </>
        ) : (
          <div className="placeholder"><div>Fields, line items and checks show up here.</div></div>
        )}
      </section>
    </div>
  );
}
