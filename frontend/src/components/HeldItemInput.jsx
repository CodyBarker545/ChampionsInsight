import React, { useEffect, useId, useState } from "react";
import { readJsonResponse, resolveApiUrl } from "../api/http.js";

let catalogRequest;

function loadCatalog() {
  if (!catalogRequest) {
    catalogRequest = fetch(resolveApiUrl("/api/items"), {
      signal: AbortSignal.timeout(8000),
    })
      .then((response) => readJsonResponse(response, "Could not load held items."))
      .then((data) => data.items || [])
      .catch((error) => {
        catalogRequest = undefined;
        throw error;
      });
  }
  return catalogRequest;
}

export default function HeldItemInput(props) {
  const listId = useId();
  const [items, setItems] = useState([]);

  useEffect(() => {
    let active = true;
    loadCatalog().then((catalog) => {
      if (active) setItems(catalog);
    }).catch(() => {});
    return () => { active = false; };
  }, []);

  return (
    <>
      <input {...props} list={listId} />
      <datalist id={listId}>
        {items.map((item) => <option key={item.name} value={item.displayName} />)}
      </datalist>
    </>
  );
}
