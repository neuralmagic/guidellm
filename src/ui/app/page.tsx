'use client';
import { Typography, useTheme } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { AppRouterCacheProvider } from '@mui/material-nextjs/v13-appRouter';
import Script from 'next/script';
import React, { ReactNode } from 'react';

import { muiThemeV3Dark } from '@/app/theme';
import { PageFooter } from '@/lib/components/PageFooter';

import { FullPageWithHeaderAndFooterLayout } from '../lib/layouts/FullPageWithHeaderAndFooterLayout';
import { ContentCenterer } from '../lib/layouts/helpers/ContentCenterer';
import { ReduxProvider } from '../lib/store/provider';

interface MyProps {
  children?: ReactNode;
}

function SafeHydrate({ children }: MyProps) {
  return <div suppressHydrationWarning>{children}</div>;
}

const Content = () => {
  const theme = useTheme();

  const header = (
    <ContentCenterer>
      <Typography color="white" variant="h1">
        Header
      </Typography>
    </ContentCenterer>
  );
  const footer = (
    <ContentCenterer>
      <PageFooter />
    </ContentCenterer>
  );
  const body = (
    <ContentCenterer>
      <Typography color="white" variant="h3">
        GuideLLM
      </Typography>
    </ContentCenterer>
  );
  return (
    <FullPageWithHeaderAndFooterLayout
      header={header}
      footer={footer}
      body={body}
      sx={{
        // TODO: instead of black, should pull from theme
        background: `linear-gradient(105deg, black, ${theme.palette.surface.surfaceContainerLowest})`,
        padding: '32px',
      }}
    />
  );
};

const Home = () => {
  return (
    <div>
      <Script id="config-script" strategy="afterInteractive"></Script>
      <ReduxProvider>
        <AppRouterCacheProvider>
          <SafeHydrate>
            <ThemeProvider theme={muiThemeV3Dark}>
              <Content />
            </ThemeProvider>
          </SafeHydrate>
        </AppRouterCacheProvider>
      </ReduxProvider>
    </div>
  );
};

export default Home;
