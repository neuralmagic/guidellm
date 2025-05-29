## Run Guidellm with Kubernetes Job

Here's an example to run `guidellm` with `meta-llama/Llama-3.2-3B-Instruct` that has been deployed with
[llm-d-deployer](https://github.com/neuralmagic/llm-d-deployer/blob/main/quickstart/README-minikube.md).
Replace the `--target` and references to `Llama-3.2-3B` in [guidellm-job.yaml](./guidellm-job.yaml) to evaluate any served LLM.

### Run evaluation

```bash
# Update the claim-name in accessor-pod.yaml, and guidellm-job.yaml if using a different pvc-name
kubectl apply -f pvc.yaml
kubectl apply -f guidellm-job.yaml
```

> **📝 NOTE:** [Dockerfile](./Dockerfile) was used to build the image for the guidellm-job pod.

> **📝 NOTE:** The HF_TOKEN is passed to the job, but this will not be necessary if you use the same PVC as the one storing your model.
> Guidellm uses the model's tokenizer/processor files in its evaluation. You can pass a path instead with `--tokenizer=/path/to/model`.
> This eliminates the need for Guidellm to download the files from Huggingface.

The logs from the job will show pretty tables that summarize the results. There is also a large yaml file created. The evaluation for this model
will take ~20-30 minutes.

### Extract Guidellm Report

```bash
kubectl apply -f accessor-pod.yaml

# Wait for the pod to be ready
kubectl wait --for=condition=Ready pod/guidellm-accessor

# Copy the report file from the pod (accessor pod mounts the volume as read-only)
kubectl cp guidellm-accessor:/app/data/guidellm-reports.tgz ./guidellm-reports.tgz
```

Extract the report:

```bash
tar -xvf guidellm-reports.tgz
```

You will now have a local file `./guidellm-reports/llama32-3b.yaml`

You can remove the accessor pod with:

```bash
kubectl delete pod guidellm-accessor
```

### Gather Insights from Guidellm Report

You can follow the ["Analyzing Results" section](../example-analysis/README.md#analyzing-results) to gain insights from your LLM
deployments using the GuideLLM report.
