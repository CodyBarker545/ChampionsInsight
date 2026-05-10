// API helpers for local RAG questions.
import { readJsonResponse, resolveApiUrl } from "./http.js";


export async function askRagQuestion(question) {
  const response = await fetch(resolveApiUrl("/api/rag/ask"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: AbortSignal.timeout(20000),
    body: JSON.stringify({ question }),
  });

  return readJsonResponse(response, "RAG question failed.");
}
