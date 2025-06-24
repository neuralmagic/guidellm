'use client';
import { Box, Link, Typography } from '@mui/material';
import React from 'react';

import { NeuralMagicTitleV2 } from '@assets/icons';

export const Component = () => {
  return (
    <Box
      component="footer"
      sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '16px',
      }}
    >
      <Box sx={{ flexGrow: 1, flexDirection: 'row' }} display="flex">
        <Typography variant="subtitle2" color="surface.onSurfaceSubdued">
          Got questions?
        </Typography>
        <Link onClick={() => alert('Not implemented')} underline="none" ml={1}>
          <Typography variant="subtitle2" color="surface.onSurfaceAccent">
            Send us a message on Slack
          </Typography>
        </Link>
      </Box>
      <Box>
        <NeuralMagicTitleV2 />
      </Box>
    </Box>
  );
};
