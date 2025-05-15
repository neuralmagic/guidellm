const nextJest = require('next/jest');

const createJestConfig = nextJest({
  dir: './',
});

const customJestConfig = {
  collectCoverage: false,
  collectCoverageFrom: ['src'],
  coverageDirectory: './.meta',
  coverageReporters: ['json-summary'],
  moduleFileExtensions: ['ts', 'tsx', 'js'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  testEnvironment: 'jest-environment-jsdom',
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest',
  },
};

module.exports = createJestConfig(customJestConfig);
