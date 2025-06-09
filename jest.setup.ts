import '@testing-library/jest-dom';

jest.mock('@nivo/bar');
jest.mock('@nivo/line');
jest.mock('@nivo/core');

jest.mock('next/dynamic', () => ({
  __esModule: true,
  default: (...props: any[]) => {
    const dynamicModule = jest.requireActual('next/dynamic');
    const dynamicActualComp = dynamicModule.default;
    const RequiredComponent = dynamicActualComp(props[0]);
    // eslint-disable-next-line no-unused-expressions, @typescript-eslint/no-unused-expressions
    RequiredComponent.preload ? RequiredComponent.preload() : RequiredComponent.render.preload();
    return RequiredComponent;
  },
}));
