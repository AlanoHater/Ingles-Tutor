import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Korean Tutor — 한국어 튜터",
  description:
    "Aprende coreano con un tutor de IA. Practica conversación, pronunciación y gramática coreana en español.",
  keywords: ["coreano", "tutor", "IA", "aprender", "hangul", "한국어"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="es">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
