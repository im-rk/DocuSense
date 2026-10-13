export const API = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000";

export type Status = "PENDING" | "PROCESSING" | "COMPLETE" | "NEEDS_REVIEW" | "FAILED";

export interface DocSummary {
  id: string;
  name: string;
  status: Status;
  error: string | null;
  created_at: string;
  finished_at: string | null;
  document_type: string | null;
  vendor: string | null;
  total: number | null;
}

export interface Word {
  text: string;
  box: [number, number, number, number];
  label: string;
  prob: number;
  line: number;
}

export interface LineItem {
  name: string;
  qty: number | null;
  price: number | null;
}

export interface Validation {
  passed: boolean;
  needs_review: boolean;
  errors: string[];
  warnings: string[];
  checks: Record<string, string>;
  missing_fields: string[];
  low_confidence_fields: string[];
}

export interface Result {
  document_type: string | null;
  type_confidence: number | null;
  type_probabilities?: Record<string, number>;
  fields: Record<string, string | number>;
  line_items: LineItem[];
  field_confidences: Record<string, number>;
  field_sources?: Record<string, string>;
  validation: Validation;
  words: Word[];
  image_size: { width: number; height: number };
  inference_ms: number | null;
}

export interface DocDetail {
  document: DocSummary;
  result: Result | null;
}

export interface Summary {
  total_documents: number;
  by_status: Record<string, number>;
  per_day: { date: string; count: number }[];
  needs_review_rate: number;
  correction_rate: number;
  total_corrections: number;
  avg_inference_ms: number | null;
  avg_confidence: Record<string, number>;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API}${path}`, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* body was not JSON */
    }
    throw new Error(msg);
  }
  return r.status === 204 ? (undefined as T) : ((await r.json()) as T);
}

export const listDocs = () => req<DocSummary[]>("/documents");
export const getDoc = (id: string) => req<DocDetail>(`/documents/${id}`);
export const deleteDoc = (id: string) => req<void>(`/documents/${id}`, { method: "DELETE" });
export const getSummary = () => req<Summary>("/analytics/summary");
export const health = () => req<{ status: string; models: { loaded: boolean; missing: string[] } }>("/health");

export function upload(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  return req<{ document_id: string; status: Status }>("/documents", { method: "POST", body: fd });
}

export const saveFields = (id: string, fields: Record<string, string | number | null>, line_items?: LineItem[]) =>
  req<DocDetail>(`/documents/${id}/fields`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fields, line_items }),
  });

export const imageUrl = (id: string) => `${API}/documents/${id}/image`;
export const exportUrl = (id: string, fmt: "json" | "csv" | "xlsx") => `${API}/documents/${id}/export?fmt=${fmt}`;
