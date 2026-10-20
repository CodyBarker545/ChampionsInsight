import React, { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { Send, X } from "lucide-react";


const SUGGESTED_QUESTIONS = [
  "What is this team trying to do?",
  "What are the likely items?",
  "What should I lead?",
  "What are the biggest threats?",
  "How do I beat this archetype?",
  "Explain Trick Room",
  "Explain Tailwind",
  "What should I watch out for?",
];


function sourceLabel(source) {
  return source === "ai" ? "AI" : "Rule-based";
}

function renderList(title, entries = [], className = "") {
  if (!entries.length) {
    return null;
  }

  return (
    <div className={`battle-advice-list ${className}`.trim()}>
      <strong>{title}</strong>
      <ul>
        {entries.map((entry) => (
          <li key={`${title}-${entry}`}>{entry}</li>
        ))}
      </ul>
    </div>
  );
}

function LoadingBubble({ label }) {
  return (
    <div className="battle-advice-assistant-shell loading" aria-live="polite">
      <div className="battle-advice-assistant-meta">
        <strong>Battle Advice</strong>
        <span className="battle-advice-source-pill thinking">Thinking</span>
      </div>
      <div className="battle-advice-assistant-bubble">
        <div className="battle-advice-typing-dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <p>{label || "Working on it..."}</p>
      </div>
    </div>
  );
}

function BattleAdviceChat({
  draftQuestion = "",
  hasTeamContext = false,
  isLoading = false,
  isOpen = false,
  messages = [],
  onClose,
  onDraftQuestionChange,
  onSubmitQuestion,
  onUseSuggestion,
}) {
  const threadRef = useRef(null);
  const threadEndRef = useRef(null);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    if (typeof threadEndRef.current?.scrollIntoView === "function") {
      threadEndRef.current.scrollIntoView({ block: "end" });
    }
  }, [isOpen, messages, isLoading]);

  if (!isOpen || typeof document === "undefined") {
    return null;
  }

  return createPortal(
    <div className="battle-advice-overlay" role="presentation" onClick={onClose}>
      <section
        className="battle-advice-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Ask Questions"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="battle-advice-modal-header">
          <div>
            <span className="battle-advice-eyebrow">Ask Questions</span>
            <h2>Battle advice, mechanics, and tournament reads</h2>
          </div>
          <div className="battle-advice-modal-actions">
            <span className="battle-advice-context-pill">
              {hasTeamContext ? "Team context attached" : "General advice only"}
            </span>
            <button
              type="button"
              className="battle-advice-close"
              onClick={onClose}
              aria-label="Close Ask Questions"
            >
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        </header>

        <div className="battle-advice-suggestions">
          {SUGGESTED_QUESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              className="battle-advice-suggestion"
              disabled={isLoading}
              onClick={() => onUseSuggestion?.(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>

        <div className="battle-advice-thread">
          <div className="battle-advice-thread-scroll" ref={threadRef}>
            {messages.length === 0 ? (
            <p className="battle-advice-empty">
              Use the top bar to ask a question. This view shows the shared chat
              history, recommended actions, and uncertainty notes from the same
              Ask Questions system.
            </p>
            ) : null}

            {messages.map((message) => {
              const result = message.result || {};
              return (
                <article className="battle-advice-message" key={message.id}>
                  <div className="battle-advice-user-bubble">{message.question}</div>

                  {message.pending ? (
                    <LoadingBubble label={message.loadingLabel} />
                  ) : (
                    <div className="battle-advice-assistant-shell">
                      <div className="battle-advice-assistant-meta">
                        <strong>{result.intent === "battle_info" ? "Battle Info" : "Battle Advice"}</strong>
                        <span className="battle-advice-source-pill">
                          {sourceLabel(result.source)}
                        </span>
                      </div>

                      <div className="battle-advice-answer-card">
                        <p className="battle-advice-answer-copy">
                          {result.answer || "No answer was returned."}
                        </p>

                        {renderList("Key points", result.keyPoints)}
                        {renderList("Recommended actions", result.recommendedActions)}
                        {renderList("Mechanics sources", result.mechanicsUsed)}
                        {renderList(
                          "Uncertainty notes",
                          result.uncertaintyNotes,
                          "warning",
                        )}
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
            <div ref={threadEndRef} />
          </div>

          <form
            className="battle-advice-composer"
            onSubmit={(event) => {
              event.preventDefault();
              onSubmitQuestion?.(draftQuestion);
            }}
          >
            <input
              aria-label="Ask Questions in chat"
              value={draftQuestion}
              placeholder="Ask a battle question..."
              onChange={(event) => onDraftQuestionChange?.(event.target.value)}
            />
            <button
              type="submit"
              disabled={isLoading || !draftQuestion.trim()}
              aria-label="Send battle question"
            >
              <Send size={16} aria-hidden="true" />
              <span>{isLoading ? "Thinking" : "Send"}</span>
            </button>
          </form>
        </div>
      </section>
    </div>,
    document.body,
  );
}

export default BattleAdviceChat;
