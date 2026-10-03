import { useEffect, useState } from "react";
import { getSummary, Summary } from "../api/client";

export default function AnalyticsPage() {
  const [s, setS] = useState<Summary | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    getSummary().then(setS).catch((e: Error) => setErr(e.message));
  }, []);
  if (err) return <div className="stats"><div className="msg bad">{err}</div></div>;
  if (!s) return <div className="stats dim">Loading…</div>;
  const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
  const max = Math.max(1, ...s.per_day.map((d) => d.count));

  return (
    <div className="stats">
      <h2>Analytics</h2>
      <div className="kpis">
        <div><b>{s.total_documents}</b><span className="dim">Documents</span></div>
        <div><b>{pct(s.needs_review_rate)}</b><span className="dim">Need review</span></div>
        <div><b>{pct(s.correction_rate)}</b><span className="dim">Corrected by hand</span></div>
        <div><b>{s.avg_inference_ms ?? "–"}</b><span className="dim">ms per document</span></div>
      </div>
      <div className="cols2">
        <div>
          <h3>Documents per day, last 14 days</h3>
          <div className="bars">
            {s.per_day.map((d) => (
              <div key={d.date} style={{ height: `${(d.count / max) * 92}%` }} title={`${d.date}: ${d.count}`}>
                {d.count > 0 && <b>{d.count}</b>}
              </div>
            ))}
          </div>
          <div className="axis">{s.per_day.map((d) => <span key={d.date}>{d.date.slice(8)}</span>)}</div>
        </div>
        <div>
          <h3>Status</h3>
          <table className="plain">
            <tbody>
              {Object.keys(s.by_status).length === 0 && <tr><td className="dim">No documents yet</td><td /></tr>}
              {Object.entries(s.by_status).map(([k, v]) => (
                <tr key={k}><td><i className={`dot ${k}`} style={{ background: undefined }} />{k.replace("_", " ").toLowerCase()}</td><td>{v}</td></tr>
              ))}
            </tbody>
          </table>
          <h3 style={{ marginTop: 22 }}>Average confidence by field</h3>
          <table className="plain">
            <tbody>
              {Object.entries(s.avg_confidence).map(([k, v]) => (
                <tr key={k}>
                  <td style={{ width: 110 }}>{k.replace("_", " ")}</td>
                  <td style={{ textAlign: "left" }}><div className="meter"><i style={{ width: pct(v) }} /></div></td>
                  <td style={{ width: 40 }}>{(v * 100).toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
