# GuideLLM UI

The GuideLLM UI is a companion to GuideLLM that allows you to visualize the peformance of your model in regards to a specific benchmark run.

## Getting Started

The two pathways to running the UI for your benchmark report are:

1. Rely on the hosted build.

If you choose this option, then you can just run your benchmarks and a report.html will be generated automatically. Assets will pull from the latest compatible version of GuideLLM UI. By default the report will be located

2. Build locally.

For this option:

First, install dependencies:

```bash
npm install
```

Then

```bash
npm run build
```

This builds the app for production to the `out` folder.\

Following this step you can serve the build however you like, for example:

```bash
npx serve out
```

Tell GuideLLM you are running things locally
