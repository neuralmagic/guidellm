import { ReactNode } from 'react';

import { useCurrentBreakpoint } from './useCurrentBreakpoint';

type HasChildren = {
  children: ReactNode;
};

const getPaddingForBreakpoint = (breakpoint: string) => {
  switch (breakpoint) {
    case 'xs':
      return '16px';
    case 'sm':
      return '32px';
    case 'md':
      return '32px';
    case 'lg':
      return '32px';
    case 'xl':
      return '32px';
    default:
      return '32px';
  }
};

export const ContentCenterer = ({ children }: HasChildren) => {
  const breakpoint = useCurrentBreakpoint();
  const padding = getPaddingForBreakpoint(breakpoint);
  const maxWidth = breakpoint === 'sm' ? '600px' : '1440px';
  return (
    <div
      style={{
        margin: '0 auto',
        padding: `0 ${padding}`,
        maxWidth: maxWidth,
        boxSizing: 'border-box',
        height: '100%',
        width: '100%',
      }}
    >
      {children}
    </div>
  );
};
