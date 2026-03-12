import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "English Tutor",
  description:
    "Aprende inglés con un tutor de IA. Practica conversación, pronunciación y gramática en inglés usando comandos de voz.",
  keywords: ["inglés", "tutor", "IA", "aprender", "english"],
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
