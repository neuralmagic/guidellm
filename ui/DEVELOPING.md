# GuideLLM UI

### Available Scripts

In the project directory, you can run:

#### `npm run dev`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

#### `npm run build`

Builds the app for production to the `out` folder.\

#### `make test-unit`

Run unit test once on your local terminal.

#### `make test-integration`

Run integration test once on your local terminal.

#### `npx cypress run --headless`

Run end to end tests against localhost:3000

##### Tagging Tests

Reference [https://www.npmjs.com/package/jest-runner-groups](jest-runner-groups)
Add @group with the tag in a docblock at the top of the test file to indicate which types of tests are contained within.
Downside is that we can't distinguish between different types of tests in the same file.

```
/**
 * Admin dashboard tests
 *
 * @group smoke
 * @group sanity
 * @group regression
 */
```

#### `make style`

Fix code styling issues.
