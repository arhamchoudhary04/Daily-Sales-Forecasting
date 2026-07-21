/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",

  // The browser calls this server same-origin; forward the API to FastAPI.
  async rewrites() {
    const backend = process.env.BACKEND_URL || "http://localhost:8000";
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      { source: "/health", destination: `${backend}/health` },
    ];
  },
};

export default nextConfig;
