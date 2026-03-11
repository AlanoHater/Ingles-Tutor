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
  "¿Cómo se dice 'hola' en coreano?",
  "Enséñame a presentarme en coreano",
  "¿Cómo cuento del 1 al 10?",
  "¿Cuáles son las vocales en hangul?",
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

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n");

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
        <span className="header-icon">🇰🇷</span>
        <h1 className="header-title">Korean Tutor</h1>
        <span className="header-subtitle">한국어 튜터</span>
        <span className="header-status" />
      </header>

      {/* Chat */}
      {messages.length === 0 ? (
        <div className="welcome">
          <span className="welcome-icon">🎓</span>
          <h2 className="welcome-title">¡Aprende coreano conmigo!</h2>
          <p className="welcome-subtitle">
            Soy tu tutor de coreano con IA. Puedes escribir en español y te
            enseñaré coreano paso a paso. ¡Prueba alguna de estas sugerencias!
          </p>
          <div className="welcome-suggestions">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                className="suggestion-chip"
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

      {/* Input */}
      <div className="input-area">
        <MicButton onTranscription={handleTranscription} />
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder="Escribe tu mensaje en español o coreano..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isLoading}
          />
        </div>
        <button
          className="btn-send"
          onClick={() => sendMessage(input)}
          disabled={!input.trim() || isLoading}
          aria-label="Enviar mensaje"
        >
          ➤
        </button>
      </div>
    </div>
  );
}
