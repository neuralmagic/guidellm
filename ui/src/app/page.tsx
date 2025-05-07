'use client';
import { useTheme } from '@mui/material';
import { useFeature } from 'lib/config/features';

import { PageHeader } from '../lib/components/PageHeader';
import { PageFooter } from '../lib/components/PageFooter';
import { WorkloadDetails } from '../lib/components/WorkloadDetails';
import { WorkloadMetrics } from '../lib/components/WorkloadMetrics';
import { CostSection } from '../lib/components/CostSection';
import { MetricsSummary } from '../lib/components/MetricsSummary';
import { ContentCenterer } from '../lib/layouts/helpers/ContentCenterer';
import { FullPageWithHeaderAndFooterLayout } from '../lib/layouts/FullPageWithHeaderAndFooterLayout';

// Save the original console.error function
const originalConsoleError = console.error;

// Override console.error to suppress errors
console.error = function () {};

// Execute code where you want to suppress errors

// Restore the original console.error after a timeout or certain operations

const Home = () => {
  setTimeout(() => {
    console.error = originalConsoleError;
  }, 10000); // Restore after 5 seconds
  const isCostSectionEnabled = useFeature('costSection');
  const theme = useTheme();

  const header = (
    <ContentCenterer>
      <PageHeader />
    </ContentCenterer>
  );
  const footer = (
    <ContentCenterer>
      <PageFooter />
    </ContentCenterer>
  );
  const body = (
    <ContentCenterer>
      <WorkloadDetails />
      <MetricsSummary />
      <WorkloadMetrics />
      {isCostSectionEnabled && <CostSection />}
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

export default Home;
