### GuideLLM UI

GuideLLM UI is a companion frontend for visualizing the results of a GuideLLM benchmark run.

### 🛠 Running the UI

1. Use the Hosted Build (Recommended for Most Users)

After running a benchmark with GuideLLM, a report.html file will be generated (by default at guidellm_report/report.html). This file references the latest stable version of the UI hosted at:

```
https://neuralmagic.github.io/guidellm/ui/dev/
```

Open the local html file in your browser and you're done—no setup required.

2. Build and Serve the UI Locally (For Development)
   This option is useful if:

- You are actively developing the UI

- You want to test changes to the UI before publishing

- You want full control over how the report is displayed

```bash
npm install
npm run build
npx serve out
```

This will start a local server (e.g., at http://localhost:3000). Then, in your GuideLLM config or CLI flags, point to this local server as the asset base for report generation.

### 🧪 Development Notes

During UI development, it can be helpful to view sample data. We include a sample benchmark run wired into the Redux store under:

```
src/lib/store/[runInfo/workloadDetails/benchmarks]WindowData.ts
```

In the future this will be replaced by a configurable untracked file for dev use.
