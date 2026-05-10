// Defines the small popup that displays RAG questions and answers.
import React from "react";
import { Send, X } from "lucide-react";


// Displays the RAG question form and grounded answer popup.
function RagPopup({
  answer,
  error,
  isAsking,
  onAsk,
  onClose,
  question,
  setQuestion,
  source,
  submittedQuestion,
}) {
  return (
    <div className="rag-overlay" role="presentation">
      <section className="rag-popup" role="dialog" aria-modal="true" aria-labelledby="rag-title">
        <button className="rag-close" type="button" onClick={onClose} aria-label="Close RAG popup">
          <X size={16} aria-hidden="true" />
        </button>
        <h2 id="rag-title">RAG Response</h2>

        <form className="rag-question-box" onSubmit={onAsk}>
          <label htmlFor="rag-question">Question:</label>
          <p id="rag-question">{submittedQuestion || question}</p>
          <div className="rag-input-row">
            <input
              aria-label="Ask another RAG question"
              value={question}
              placeholder="Ask another question"
              onChange={(event) => setQuestion(event.target.value)}
            />
            <button className="icon-button" type="submit" disabled={isAsking} aria-label="Ask RAG question">
              <Send size={16} aria-hidden="true" />
            </button>
          </div>
        </form>

        <div className="rag-answer-box">
          <strong>Response:</strong>
          {error ? (
            <p className="rag-error">{error}</p>
          ) : (
            <p>{answer || "Submit a RAG question to see a grounded answer."}</p>
          )}
        </div>

        <p className="rag-source">Source: {source || "Local RAG Knowledge Base"}</p>
      </section>
    </div>
  );
}


export default RagPopup;
