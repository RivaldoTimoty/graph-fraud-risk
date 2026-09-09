import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Vercel menangani Next.js secara native, jadi tidak perlu output: 'export'.
  // trailingSlash juga dilepas: kombinasinya dengan static export membuat Vercel
  // mencari halaman di out/index/index.html sementara export menaruhnya di
  // out/index.html, dan ketidakcocokan itu memunculkan 404.
  //
  // Untuk deploy statis ke host lain (GitHub Pages, Netlify static), tambahkan
  // kembali output: "export" lalu build menghasilkan folder out/.
  images: { unoptimized: true },
  // Jangan buat AGENTS.md/CLAUDE.md di dashboard/ - project sudah punya CLAUDE.md
  // sendiri di root, dan file duplikat di subfolder membingungkan.
  agentRules: false,
};

export default nextConfig;
