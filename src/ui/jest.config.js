const nextJest = require('next/jest');

const createJestConfig = nextJest({
  dir: __dirname,
});

const customJestConfig = {
  collectCoverage: false,
  collectCoverageFrom: ['tests'],
  coverageDirectory: './.meta',
  coverageReporters: ['json-summary'],
  moduleFileExtensions: ['ts', 'tsx', 'js'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  testEnvironment: 'jest-environment-jsdom',
};

module.exports = createJestConfig(customJestConfig);
