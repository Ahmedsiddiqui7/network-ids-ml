import type { Metadata } from "next";
import { IBM_Plex_Sans_Condensed, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const sans = IBM_Plex_Sans_Condensed({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const mono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "NIDS Replay Dashboard",
  description: "Replayed flow predictions from the AI-driven network intrusion detection API.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} h-full`}>
      <body className="flex min-h-full flex-col bg-bg font-sans text-text-primary antialiased">
        {children}
      </body>
    </html>
  );
}
