// Shared HTTP helpers for frontend API modules.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export function resolveApiUrl(pathOrUrl = "") {
  if (!pathOrUrl) {
    return "";
  }

  if (/^https?:\/\//i.test(pathOrUrl) || pathOrUrl.startsWith("blob:")) {
    return pathOrUrl;
  }

  const normalizedPath = pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`;
  return `${API_BASE_URL}${normalizedPath}`;
}

export async function readJsonResponse(response, fallbackMessage) {
  const result = await response.json();

  if (!response.ok) {
    throw new Error(result.error ?? fallbackMessage);
  }

  return result;
}
