"use client";

import { useState, useRef, useCallback } from "react";

interface MicButtonProps {
  onTranscription: (text: string) => void;
}

export default function MicButton({ onTranscription }: MicButtonProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });

      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Detener tracks del micrófono
        stream.getTracks().forEach((track) => track.stop());

        const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" });
        if (audioBlob.size === 0) return;

        setIsProcessing(true);

        try {
          const formData = new FormData();
          formData.append("audio", audioBlob, "recording.webm");

          const backendUrl =
            process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
          const response = await fetch(`${backendUrl}/asr`, {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`ASR error: ${response.status}`);
          }

          const data = await response.json();
          if (data.text?.trim()) {
            onTranscription(data.text.trim());
          }
        } catch (error) {
          console.error("Error en ASR:", error);
        } finally {
          setIsProcessing(false);
        }
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
    } catch (error) {
      console.error("Error accediendo al micrófono:", error);
    }
  }, [onTranscription]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, []);

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <button
      className={`btn-mic ${isRecording ? "recording" : ""}`}
      onClick={toggleRecording}
      disabled={isProcessing}
      aria-label={isRecording ? "Detener grabación" : "Grabar audio"}
      title={
        isProcessing
          ? "Procesando..."
          : isRecording
          ? "Click para detener"
          : "Mantén para hablar en coreano"
      }
    >
      {isProcessing ? "⏳" : isRecording ? "⏹" : "🎤"}
    </button>
  );
}
