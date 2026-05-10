// Defines a reusable card for one analysis result.
import React from "react";


// Displays one analysis result section.
function ResultCard({ children, icon, title }) {
  return (
    <article className="result-card">
      <div className="result-title">
        {icon}
        <h3>{title}</h3>
      </div>
      {children}
    </article>
  );
}


export default ResultCard;
