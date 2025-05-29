## Getting Started

The GuideLLM UI is built with Next.js

First, install dependencies:

```bash
npm install
```

Then, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the app.

#### `npm run build`

Builds the app for production to the `out` folder.\

#### `make test-unit`

Run unit test once on your local terminal.

#### `make test-integration`

Run integration test once on your local terminal.

#### `npx cypress run --headless`

Run end to end tests against localhost:3000

#### `make style`

Fix code styling issues.

#### `make quality`

Run quality eslint quality checks.

##### Tagging Tests

Reference [https://www.npmjs.com/package/jest-runner-groups](jest-runner-groups)
Add @group with the tag in a docblock at the top of the test file to indicate which types of tests are contained within.
Can't distinguish between different types of tests in the same file.

```
/**
 * Admin dashboard tests
 *
 * @group smoke
 * @group sanity
 * @group regression
 */
```
