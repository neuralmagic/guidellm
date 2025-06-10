// @ts-check

import eslint from '@eslint/js';
import typescriptPlugin from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';
import { FlatCompat } from '@eslint/eslintrc';
import reactPlugin from 'eslint-plugin-react';
import hooksPlugin from 'eslint-plugin-react-hooks';
import importPlugin from 'eslint-plugin-import';
import jestPlugin from 'eslint-plugin-jest';
import noSecretsPlugin from 'eslint-plugin-no-secrets';
import prettierPlugin from 'eslint-plugin-prettier';
import prettierConfig from 'eslint-config-prettier';
import globals from 'globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: eslint.configs.recommended,
});

export default [
  // Base configuration
  eslint.configs.recommended,

  // Next.js configuration using FlatCompat
  ...compat.extends('next/core-web-vitals'),

  // --- Main Configuration for your files ---
  {
    files: ['src/**/*.{js,jsx,ts,tsx}', 'tests/**/*.{js,jsx,ts,tsx}'],
    languageOptions: {
      parser: typescriptParser,
      ecmaVersion: 2024,
      sourceType: 'module',
      globals: {
        ...globals.browser,
        ...globals.node,
        ...globals.jest,
      },
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
        project: [
          './src/ui/tsconfig.json',
          './tsconfig.test.json',
          './tsconfig.cypress.json',
        ],
        tsconfigRootDir: import.meta.dirname,
        noWarnOnMultipleProjects: true,
      },
    },
    plugins: {
      '@typescript-eslint': typescriptPlugin,
      react: reactPlugin,
      'react-hooks': hooksPlugin,
      import: importPlugin,
      jest: jestPlugin,
      'no-secrets': noSecretsPlugin,
      prettier: prettierPlugin,
    },
    rules: {
      // Your custom rules
      complexity: ['warn', { max: 8 }],
      curly: ['error', 'all'],
      'no-unused-vars': 'off',

      // TypeScript rules
      '@typescript-eslint/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      '@typescript-eslint/no-explicit-any': 'warn',

      // Next.js overrides (these will override the ones from next/core-web-vitals)
      '@next/next/no-img-element': 'off', // Allow img tags if needed
      '@next/next/no-page-custom-font': 'warn',

      // React rules
      'react/react-in-jsx-scope': 'off', // Not needed in Next.js
      'react/prop-types': 'off', // Using TypeScript
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',

      // Import rules
      'import/no-extraneous-dependencies': [
        'error',
        {
          devDependencies: [
            '**/*.test.{js,jsx,ts,tsx}',
            '**/*.d.ts',
            '**/*.interfaces.ts',
            '**/*.setup.{js,ts}',
            '**/*.config.{js,mjs,ts}',
            'tests/**/*',
            'cypress/**/*',
          ],
          optionalDependencies: false,
          peerDependencies: false,
        },
      ],
      'import/order': [
        'error',
        {
          groups: [
            ['builtin', 'external'],
            ['internal', 'parent', 'sibling', 'index'],
          ],
          'newlines-between': 'always-and-inside-groups',
          pathGroups: [
            {
              pattern:
                '@{app,assets,classes,components,hooks,lib,pages,store,tests,types,utils}/**',
              group: 'internal',
              position: 'before',
            },
            {
              pattern: '{.,..}/**',
              group: 'internal',
              position: 'after',
            },
          ],
          pathGroupsExcludedImportTypes: ['builtin'],
          alphabetize: { order: 'asc', caseInsensitive: true },
        },
      ],

      // Security
      'no-secrets/no-secrets': ['error', { additionalRegexes: {}, ignoreContent: [] }],

      // Prettier
      'prettier/prettier': 'error',
    },
    settings: {
      next: {
        rootDir: ['src/ui/', 'tests/ui/'],
      },
      'import/resolver': {
        typescript: {
          project: [
            './src/ui/tsconfig.json',
            './tsconfig.test.json',
            './tsconfig.cypress.json',
          ],
          noWarnOnMultipleProjects: true,
        },
      },
      react: {
        version: 'detect',
      },
    },
  },

  // Jest-specific rules for test files
  {
    files: [
      'tests/**/*.{js,jsx,ts,tsx}',
      '**/*.test.{js,jsx,ts,tsx}',
      '**/*.spec.{js,jsx,ts,tsx}',
    ],
    rules: {
      'jest/expect-expect': 'error',
      'jest/no-focused-tests': 'error',
      'jest/no-identical-title': 'error',
      'jest/prefer-to-have-length': 'warn',
      'jest/valid-expect': 'error',
    },
  },

  // Prettier config (disables conflicting rules)
  prettierConfig,
];
