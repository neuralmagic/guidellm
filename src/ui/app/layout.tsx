import type { Metadata } from 'next';
import React from 'react';

import './globals.css';

export const metadata: Metadata = {
  title: 'GuideLLM',
  description: 'LLM Benchmarking Tool',
};

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
      </head>
      <body>{children}</body>
    </html>
  );
}
