import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone", // a self-contained server for the production image
  cacheComponents: true,
  partialPrefetching: true,
};

export default nextConfig;
