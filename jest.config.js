const nextJest = require('next/jest');

const createJestConfig = nextJest({
  dir: './src/ui',
});

const customJestConfig = {
  collectCoverage: true,
  collectCoverageFrom: ['./src/ui/**/*.{ts,tsx}'],
  coverageDirectory: './coverage',
  coverageProvider: 'v8',
  coverageReporters: ['json', 'text-summary', 'lcov'],
  moduleFileExtensions: ['ts', 'tsx', 'js'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  testEnvironment: 'jest-environment-jsdom',
  testMatch: [
    '<rootDir>/tests/ui/unit/**/*.(test|spec).{ts,tsx,js,jsx}',
    '<rootDir>/tests/ui/integration/**/*.(test|spec).{ts,tsx,js,jsx}',
  ],
};

module.exports = createJestConfig(customJestConfig);
