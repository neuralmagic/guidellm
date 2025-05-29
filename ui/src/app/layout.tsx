import React from 'react';

import { benchmarksScript } from '@/lib/store/benchmarksWindowData';
import { runInfoScript } from '@/lib/store/runInfoWindowData';
import { workloadDetailsScript } from '@/lib/store/workloadDetailsWindowData';

import './globals.css';

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const assetPrefix = process.env.ASSET_PREFIX || '';

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <link rel="icon" href={`${assetPrefix}/favicon.ico`} />
        <link rel="apple-touch-icon" href={`${assetPrefix}/favicon-192x192.png`} />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#000000" />
        <meta name="description" content="LLM Benchmarking Tool" />
        <meta name="title" content="GuideLLM" />
        <link rel="manifest" href={`${assetPrefix}/manifest.json`} />
        <title>Guide LLM</title>
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
