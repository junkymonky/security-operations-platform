/** @type {import('next').NextConfig} */
const ML_API_URL = process.env.ML_API_URL || 'http://127.0.0.1:8000'

const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  experimental: {
    // Model training on large datasets can exceed the default 30 s proxy timeout ("socket hang up").
    proxyTimeout: 300_000,
  },
  // Proxy the Python ML service through the Next.js origin (no CORS needed).
  async rewrites() {
    return [{ source: '/ml-api/:path*', destination: `${ML_API_URL}/:path*` }]
  },
}

export default nextConfig
