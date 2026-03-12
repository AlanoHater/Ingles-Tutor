"use client";

import { useState, useRef, useCallback } from "react";
import ChatBox from "@/components/ChatBox";
import MicButton from "@/components/MicButton";
import AudioPlayer from "@/components/AudioPlayer";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

const SUGGESTIONS = [
  "How do I say 'hello' in English?",
  "Teach me how to introduce myself",
  "How do I order a coffee in a restaurant?",
  "What is the difference between 'in', 'on', and 'at'?",
];

// ---------------------------------------------------------------------------
// Page Component
// ---------------------------------------------------------------------------
export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // ---- Send message ----
  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isLoading) return;

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: text.trim(),
      };

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setInput("");
      setIsLoading(true);

      try {
        const allMessages = [...messages, userMsg];

        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: allMessages.map((m) => ({
              role: m.role,
              content: m.content,
            })),
          }),
        });

        if (!response.ok) {
          throw new Error(`Error ${response.status}: ${response.statusText}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No se pudo leer la respuesta");

        const decoder = new TextDecoder();
        let fullContent = "";
        let buffer = ""; // Buffer para chunks fragmentados

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          
          // El último elemento puede ser un fragmento incompleto, lo guardamos en el buffer
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6).trim();
              if (data === "[DONE]") continue;

              try {
                const parsed = JSON.parse(data);
                const delta = parsed.choices?.[0]?.delta?.content;
                if (delta) {
                  fullContent += delta;
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMsg.id
                        ? { ...m, content: fullContent }
                        : m
                    )
                  );
                }
              } catch {
                // skip malformed chunks
              }
            }
          }
        }
      } catch (error) {
        console.error("Error en chat:", error);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? {
                  ...m,
                  content:
                    "⚠️ Error conectando con el tutor. Verifica que el backend esté corriendo.",
                }
              : m
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [messages, isLoading]
  );

  // ---- Handle keyboard ----
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  // ---- Handle ASR result ----
  const handleTranscription = (text: string) => {
    setInput(text);
    inputRef.current?.focus();
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <span className="header-icon">🇺🇸</span>
          <div className="header-text">
            <h1 className="header-title">English Tutor</h1>
            <span className="header-subtitle">Your AI Teacher</span>
          </div>
        </div>
        <div className="header-status-pill">
          <span className="status-dot" /> Listo
        </div>
      </header>

      {/* Chat Area */}
      {messages.length === 0 ? (
        <div className="welcome">
          <span className="welcome-avatar">👋</span>
          <h2 className="welcome-title">¡Hola! Hello!</h2>
          <p className="welcome-subtitle">
            I am your personal English tutor. I'm here to help you improve 
            your vocabulary, grammar, and conversational fluency. What would you like to practice today?
          </p>
          <div className="suggestions-grid">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                className="suggestion-card"
                onClick={() => sendMessage(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <ChatBox messages={messages} isLoading={isLoading} />
      )}

      {/* Input Area */}
      <div className="input-container">
        <div className="input-box">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder="Escribe en español o inglés..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isLoading}
          />
          <div className="fab-container">
            <MicButton onTranscription={handleTranscription} />
            <button
              className="btn-icon btn-send"
              onClick={() => sendMessage(input)}
              disabled={!input.trim() || isLoading}
              aria-label="Enviar mensaje"
            >
              ➤
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
