import { render, screen } from '@testing-library/react';

import Home from '../../app/page';
import { MockedWrapper } from '../test.helper';

test('renders app name', async () => {
  render(
    <MockedWrapper>
      <Home />
    </MockedWrapper>
  );
  const appElement = screen.getByText(/Guidellm/i);
  expect(appElement).toBeInTheDocument();
});
