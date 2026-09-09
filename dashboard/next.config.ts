import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export - tidak butuh server Node saat runtime, deploy ke Vercel/Pages
  // tinggal connect repo.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
  // Jangan buat AGENTS.md/CLAUDE.md di dashboard/ - project sudah punya CLAUDE.md
  // sendiri di root, dan file duplikat di subfolder membingungkan.
  agentRules: false,
};

export default nextConfig;
