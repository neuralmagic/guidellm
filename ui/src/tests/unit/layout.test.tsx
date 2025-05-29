import { render } from '@testing-library/react';

import RootLayout from '@/app/layout';

describe('RootLayout', () => {
  it('renders children inside the layout', () => {
    const { getByText } = render(
      <RootLayout>
        <p>Test Child</p>
      </RootLayout>
    );

    expect(getByText('Test Child')).toBeInTheDocument();
  });
});
