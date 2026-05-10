// Defines the API connection status indicator.
import React from "react";
import { Activity } from "lucide-react";


// Displays the current API connection status.
function StatusPill({ status }) {
  return (
    <div className={`status-pill ${status}`}>
      <Activity size={16} aria-hidden="true" />
      <span>API {status}</span>
    </div>
  );
}


export default StatusPill;
