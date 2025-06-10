import type { Metadata, Viewport } from 'next';
import React from 'react';

import { benchmarksScript } from '@/lib/store/benchmarksWindowData';
import { runInfoScript } from '@/lib/store/runInfoWindowData';
import { workloadDetailsScript } from '@/lib/store/workloadDetailsWindowData';
import './globals.css';

export async function generateMetadata(): Promise<Metadata> {
  const assetPrefix = process.env.ASSET_PREFIX || '';

  return {
    title: 'GuideLLM',
    description: 'LLM Benchmarking Tool',
    icons: {
      icon: `${assetPrefix}/favicon.png`,
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
      <head>
        {/* <script
          dangerouslySetInnerHTML={{
            __html: 'window.run_info = {}; window.workload_details = {}; window.benchmarks = {};',
          }}
        /> */}
        <script
          dangerouslySetInnerHTML={{
            __html: runInfoScript,
          }}
        />
        <script
          dangerouslySetInnerHTML={{
            __html: workloadDetailsScript,
          }}
        />
        <script
          dangerouslySetInnerHTML={{
            __html: benchmarksScript,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
