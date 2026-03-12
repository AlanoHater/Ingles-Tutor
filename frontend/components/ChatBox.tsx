"use client";

import { useEffect, useRef } from "react";
import AudioPlayer from "./AudioPlayer";
import type { Message } from "@/app/page";

interface ChatBoxProps {
  messages: Message[];
  isLoading: boolean;
}

export default function ChatBox({ messages, isLoading }: ChatBoxProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll al nuevo mensaje
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-area">
      {messages.map((msg) => (
        <div key={msg.id} className={`message message--${msg.role}`}>
          {msg.role === "assistant" && (
            <span className="message-label">🇰🇷 Tutor</span>
          )}
          <div className="message-bubble">
            {msg.content || (msg.role === "assistant" && isLoading ? "" : "")}
          </div>
          {msg.role === "assistant" && msg.content && (
            <div className="message-actions">
              <AudioPlayer text={msg.content} />
            </div>
          )}
        </div>
      ))}

      {/* Typing indicator */}
      {isLoading &&
        messages.length > 0 &&
        !messages[messages.length - 1].content && (
          <div className="message message--tutor">
            <span className="message-label">🇰🇷 Tutor</span>
            <div className="typing-indicator">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </div>
        )}

      <div ref={bottomRef} />
    </div>
  );
}
