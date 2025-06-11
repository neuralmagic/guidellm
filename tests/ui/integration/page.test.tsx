import { render } from '@testing-library/react';

import Home from '@/app/page';

describe('Home Page', () => {
  it('renders the homepage content', () => {
    const { getByText } = render(<Home />);
    expect(getByText('GuideLLM')).toBeInTheDocument();
  });
});
