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
  const responseText = await response.text();
  let result = null;

  if (responseText) {
    try {
      result = JSON.parse(responseText);
    } catch {
      const isHtmlResponse = /^\s*</.test(responseText);
      const message =
        isHtmlResponse && response.status === 404
          ? `${fallbackMessage} The backend route may be unavailable.`
          : `${fallbackMessage} The server returned a non-JSON response.`;

      throw new Error(message);
    }
  }

  if (!response.ok) {
    throw new Error(result?.error ?? fallbackMessage);
  }

  return result ?? {};
}
