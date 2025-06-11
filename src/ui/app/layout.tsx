import type { Metadata, Viewport } from 'next';
import React from 'react';

import './globals.css';

export async function generateMetadata(): Promise<Metadata> {
  const assetPrefix = process.env.ASSET_PREFIX || '';

  return {
    title: 'GuideLLM',
    description: 'LLM Benchmarking Tool',
    icons: {
      icon: `${assetPrefix}/favicon.ico`,
      apple: `${assetPrefix}/favicon-192x192.png`,
    },
    manifest: `${assetPrefix}/manifest.json`,
  };
}

export const viewport: Viewport = {
  initialScale: 1,
  themeColor: '#000000',
  width: 'device-width',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
