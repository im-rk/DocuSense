import { useCallback, useEffect, useRef, useState } from "react";
import { health, upload } from "./api/client";
import AnalyticsPage from "./pages/AnalyticsPage";
import Workspace from "./pages/Workspace";

type Models = { loaded: boolean; missing: string[] } | "down" | null;

export default function App() {
  const [tab, setTab] = useState<"docs" | "stats">("docs");
  const [selected, setSelected] = useState<string | null>(null);
  const [models, setModels] = useState<Models>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [msg, setMsg] = useState<{ text: string; bad: boolean } | null>(null);
  const file = useRef<HTMLInputElement>(null);
  const timer = useRef<number>();

  const toast = useCallback((text: string, bad = false) => {
    setMsg({ text, bad });
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setMsg(null), 5000);
  }, []);

  useEffect(() => {
    health().then((h) => setModels(h.models)).catch(() => setModels("down"));
  }, [refreshKey]);

  const send = useCallback(
    async (files: FileList | File[]) => {
      let last: string | null = null;
      for (const f of Array.from(files)) {
        try {
          last = (await upload(f)).document_id;
        } catch (e) {
          toast(`${f.name}: ${(e as Error).message}`, true);
        }
      }
      setTab("docs");
      setRefreshKey((k) => k + 1);
      if (last) setSelected(last);
    },
    [toast],
  );

  // drop files anywhere on the window
  useEffect(() => {
    let depth = 0;
    const has = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
    const enter = (e: DragEvent) => { if (has(e)) { depth++; setDragging(true); } };
    const leave = (e: DragEvent) => { if (has(e) && --depth <= 0) { depth = 0; setDragging(false); } };
    const over = (e: DragEvent) => { if (has(e)) e.preventDefault(); };
    const drop = (e: DragEvent) => {
      if (!has(e)) return;
      e.preventDefault();
      depth = 0;
      setDragging(false);
      if (e.dataTransfer?.files.length) void send(e.dataTransfer.files);
    };
    const clear = () => { depth = 0; setDragging(false); };       // mouse events never fire mid-drag, so this un-sticks a cancelled drag
    window.addEventListener("mousemove", clear);
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("mousemove", clear);
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("dragover", over);
      window.removeEventListener("drop", drop);
    };
  }, [send]);

  const state = models === null ? "" : models === "down" || models.missing.length ? "bad" : "ok";
  const text =
    models === null ? "Connecting…" : models === "down" ? "API not reachable" : models.missing.length ? `Missing: ${models.missing.join(", ")}` : "Models ready";

  return (
    <div className="app">
      <header className="top">
        <span className="name">DocuSense</span>
        <button className={`tab ${tab === "docs" ? "on" : ""}`} onClick={() => setTab("docs")}>Documents</button>
        <button className={`tab ${tab === "stats" ? "on" : ""}`} onClick={() => setTab("stats")}>Analytics</button>
        <span className="sp" />
        <span className="status"><i className={`dot ${state}`} /><b>{text}</b></span>
        <button className="btn" onClick={() => file.current?.click()}>Upload</button>
        <input
          ref={file}
          type="file"
          multiple
          hidden
          accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp"
          onChange={(e) => {
            if (e.target.files) void send(e.target.files);
            e.target.value = "";
          }}
        />
      </header>
      {tab === "docs" ? (
        <Workspace selectedId={selected} onSelect={setSelected} refreshKey={refreshKey} toast={toast} />
      ) : (
        <AnalyticsPage />
      )}
      {dragging && <div className="drop">Drop to upload</div>}
      {msg && <div className={`toast ${msg.bad ? "bad" : ""}`}>{msg.text}</div>}
    </div>
  );
}
