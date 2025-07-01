'use client';
import { Box, Link, Typography, useTheme } from '@mui/material';
import dynamic from 'next/dynamic';
import NextLink from 'next/link';

import { Open } from '@assets/icons';

import { useGetRunInfoQuery } from '../../store/slices/runInfo';
import { formateDate, getFileSize } from '../../utils/helpers';
import { SvgContainer } from '../../utils/SvgContainer';
import { SpecBadge } from '../SpecBadge';
import { HeaderCell, HeaderWrapper } from './PageHeader.styles';

const Component = () => {
  const theme = useTheme();
  const { data } = useGetRunInfoQuery();
  const modelSize = getFileSize(data?.model?.size || 0);

  return (
    <Box py={2}>
      <Typography variant="subtitle2" color="surface.onSurfaceAccent">
        GuideLLM
      </Typography>
      <Typography variant="h4" color="surface.onSurface" my={'12px'}>
        Workload Report
      </Typography>
      <HeaderWrapper container>
        <HeaderCell item xs={5} withDivider sx={{ paddingLeft: 0 }}>
          <SpecBadge
            label="Model"
            value={data?.model?.name || 'N/A'}
            variant="metric2"
            withTooltip
          />
          <SpecBadge
            label="Model size"
            value={data?.model?.size ? `${modelSize?.size} ${modelSize?.units}` : '0B'}
            variant="body1"
          />
        </HeaderCell>
        {/*<HeaderCell item xs={2} withDivider>*/}
        {/*  <SpecBadge*/}
        {/*    label="Hardware"*/}
        {/*    value="A10"*/}
        {/*    variant="metric2"*/}
        {/*    additionalValue={*/}
        {/*      <Chip*/}
        {/*        label="x3"*/}
        {/*        color="primary"*/}
        {/*        variant="filled"*/}
        {/*        size="small"*/}
        {/*        sx={{ padding: '0 5px' }}*/}
        {/*      />*/}
        {/*    }*/}
        {/*    key="Hardware"*/}
        {/*  />*/}
        {/*  <SpecBadge label="Interconnect" value="PCIE" variant="body1" key="Interconnect" />*/}
        {/*  <SpecBadge*/}
        {/*    label="Version"*/}
        {/*    value={reportData.server.hardware_driver}*/}
        {/*    variant="body1"*/}
        {/*    key="Version"*/}
        {/*  />*/}
        {/*</HeaderCell>*/}
        <HeaderCell item xs={2} withDivider>
          <SpecBadge label="Task" value={data?.task || 'n/a'} variant="metric2" />
        </HeaderCell>
        <HeaderCell item xs={3} withDivider>
          <SpecBadge
            label="Dataset"
            value={data?.dataset?.name || 'n/a'}
            variant="metric2"
            additionalValue={
              <Link href="https://example.com" target="_blank" component={NextLink}>
                <SvgContainer color={theme.palette.primary.main}>
                  <Open />
                </SvgContainer>
              </Link>
            }
          />
        </HeaderCell>
        <HeaderCell item xs={2} sx={{ paddingRight: 0 }}>
          <SpecBadge
            label="Time Stamp"
            value={data?.timestamp ? formateDate(data?.timestamp) : 'n/a'}
            variant="caption"
          />
        </HeaderCell>
      </HeaderWrapper>
    </Box>
  );
};

const DynamicComponent = dynamic(() => Promise.resolve(Component), {
  ssr: false,
});

export { DynamicComponent as Component };
