"use client";

import { useState, useRef, useCallback, useEffect } from "react";

interface MicButtonProps {
  onTranscription: (text: string) => void;
}

export default function MicButton({ onTranscription }: MicButtonProps) {
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    // Initialize SpeechRecognition
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = "en-US"; // English recognition

      recognition.onstart = () => {
        setIsRecording(true);
      };

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        if (transcript.trim()) {
          onTranscription(transcript.trim());
        }
      };

      recognition.onerror = (event: any) => {
        console.error("Speech recognition error", event.error);
        setIsRecording(false);
      };

      recognition.onend = () => {
        setIsRecording(false);
      };

      recognitionRef.current = recognition;
    } else {
      console.error("Browser does not support Speech Recognition");
    }
  }, [onTranscription]);

  const toggleRecording = useCallback(() => {
    if (!recognitionRef.current) return;

    if (isRecording) {
      recognitionRef.current.stop();
      setIsRecording(false);
    } else {
      try {
        recognitionRef.current.start();
      } catch (err) {
        console.error(err);
      }
    }
  }, [isRecording]);

  if (!recognitionRef.current && typeof window !== 'undefined') {
    return (
      <button className="btn-mic" disabled title="Not supported in this browser">
        🎙️
      </button>
    );
  }

  return (
    <button
      className={`btn-mic ${isRecording ? "recording" : ""}`}
      onClick={toggleRecording}
      aria-label={isRecording ? "Stop recording" : "Record audio in English"}
      title={isRecording ? "Click to stop" : "Speak in English"}
    >
      {isRecording ? "⏹" : "🎤"}
    </button>
  );
}
