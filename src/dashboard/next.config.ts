import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Silences Turbopack's workspace-root auto-detection warning: it finds an
  // unrelated package-lock.json at $HOME (outside this repo) and would
  // otherwise guess that as the root.
  turbopack: {
    root: path.join(__dirname),
  },
  // Next.js 16 auto-generates AGENTS.md/CLAUDE.md dev-tooling files on
  // `next dev` -- this repo already has its own root CLAUDE.md.
  agentRules: false,
};

export default nextConfig;
