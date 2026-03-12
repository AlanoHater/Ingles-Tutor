"use client";

import { useState, useRef, useCallback } from "react";

interface AudioPlayerProps {
  text: string;
}

export default function AudioPlayer({ text }: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Extract only the English part if the formatted "Inglés: [...]" is found
  const extractEnglish = (content: string): string => {
    const englishMatch = content.match(/Inglés:\s*(.+)/i);
    return englishMatch ? englishMatch[1].trim() : content;
  };

  const playAudio = useCallback(async () => {
    if (isPlaying && audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
      return;
    }

    const englishText = extractEnglish(text);
    if (!englishText.trim()) return;

    setIsLoading(true);

    try {
      const backendUrl =
        process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const response = await fetch(`${backendUrl}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: englishText }),
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
