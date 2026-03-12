"use client";

import { useState, useRef, useCallback } from "react";

interface AudioPlayerProps {
  text: string;
}

export default function AudioPlayer({ text }: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Extraer solo texto coreano para TTS (caracteres hangul)
  const extractKorean = (content: string): string => {
    const koreanRegex = /[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]+/g;
    const matches = content.match(koreanRegex);
    return matches ? matches.join(" ") : content;
  };

  const playAudio = useCallback(async () => {
    if (isPlaying && audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
      return;
    }

    const koreanText = extractKorean(text);
    if (!koreanText.trim()) return;

    setIsLoading(true);

    try {
      const backendUrl =
        process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const response = await fetch(`${backendUrl}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: koreanText }),
      });

      if (!response.ok) {
        throw new Error(`TTS error: ${response.status}`);
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      const audio = new Audio(audioUrl);
      audioRef.current = audio;

      audio.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
      };

      audio.onerror = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();
      setIsPlaying(true);
    } catch (error) {
      console.error("Error en TTS:", error);
    } finally {
      setIsLoading(false);
    }
  }, [text, isPlaying]);

  return (
    <button
      className={`btn-action audio-btn ${isPlaying ? "playing" : ""}`}
      onClick={playAudio}
      disabled={isLoading}
      aria-label={isPlaying ? "Detener audio" : "Escuchar pronunciación"}
    >
      {isLoading ? (
        "⏳"
      ) : isPlaying ? (
        <>
          ⏹
          <span className="audio-visualizer">
            <span className="audio-bar" />
            <span className="audio-bar" />
            <span className="audio-bar" />
            <span className="audio-bar" />
          </span>
        </>
      ) : (
        "🔊 Escuchar"
      )}
    </button>
  );
}
