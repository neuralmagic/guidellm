'use client';
import Script from 'next/script';
import { ThemeProvider } from '@mui/material/styles';
import './globals.css';
import React, { ReactNode } from 'react';
import { AppRouterCacheProvider } from '@mui/material-nextjs/v13-appRouter';
import { muiThemeV3Dark } from 'app/theme';

import { ReduxProvider } from '../lib/store/provider';
import { runInfoScript } from '../lib/store/runInfoWindowData';
import { workloadDetailsScript } from '../lib/store/workloadDetailsWindowData';
import { benchmarksScript } from '../lib/store/benchmarksWindowData';

interface MyProps {
  children?: ReactNode;
}

function SafeHydrate({ children }: MyProps) {
  return <div suppressHydrationWarning>{children}</div>;
}

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
        <meta name="description" content="Guidellm" />
        {/*
          manifest.json provides metadata used when your web app is installed on a
          user's mobile device or desktop. See https://developers.google.com/web/fundamentals/web-app-manifest/
        */}
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
      <body>
        <Script id="config-script" strategy="afterInteractive"></Script>
        <ReduxProvider>
          <AppRouterCacheProvider>
            <SafeHydrate>
              <ThemeProvider theme={muiThemeV3Dark}>{children}</ThemeProvider>
            </SafeHydrate>
          </AppRouterCacheProvider>
        </ReduxProvider>
      </body>
    </html>
  );
}
