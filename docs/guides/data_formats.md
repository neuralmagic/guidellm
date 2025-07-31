# Data Formats

The `--data` argument for the `guidellm benchmark run` command accepts several different formats for specifying the data to be used for benchmarking.

## Local Data Files

You can provide a path to a local data file in one of the following formats:

- **CSV (.csv)**: A comma-separated values file. The loader will attempt to find a column with a common name for the prompt (e.g., `prompt`, `text`, `instruction`).
- **JSON (.json)**: A JSON file. The structure should be a list of objects, where each object represents a row of data.
- **JSON Lines (.jsonl)**: A file where each line is a valid JSON object.
- **Text (.txt)**: A plain text file, where each line is treated as a separate prompt.

If the prompt column cannot be automatically determined, you can specify it using the `--data-args` option:

```bash
--data-args '{"text_column": "my_custom_prompt_column"}'
```

## Synthetic Data

You can generate synthetic data on the fly by providing a configuration string or file.

### Configuration Options

| Parameter             | Description                                                                                                     |
| --------------------- | --------------------------------------------------------------------------------------------------------------- |
| `prompt_tokens`       | **Required.** The average number of tokens for the generated prompts.                                           |
| `output_tokens`       | **Required.** The average number of tokens for the generated outputs.                                           |
| `samples`             | The total number of samples to generate. Defaults to 1000.                                                      |
| `source`              | The source text to use for generating the synthetic data. Defaults to a built-in copy of "Pride and Prejudice". |
| `prompt_tokens_stdev` | The standard deviation of the tokens generated for prompts.                                                     |
| `prompt_tokens_min`   | The minimum number of text tokens generated for prompts.                                                        |
| `prompt_tokens_max`   | The maximum number of text tokens generated for prompts.                                                        |
| `output_tokens_stdev` | The standard deviation of the tokens generated for outputs.                                                     |
| `output_tokens_min`   | The minimum number of text tokens generated for outputs.                                                        |
| `output_tokens_max`   | The maximum number of text tokens generated for outputs.                                                        |

### Configuration Formats

You can provide the synthetic data configuration in one of three ways:

1. **Key-Value String:**

   ```bash
   --data "prompt_tokens=256,output_tokens=128,samples=500"
   ```

2. **JSON String:**

   ```bash
   --data '{"prompt_tokens": 256, "output_tokens": 128, "samples": 500}'
   ```

3. **YAML or Config File:** Create a file (e.g., `my_config.yaml`):

   ```yaml
   prompt_tokens: 256
   output_tokens: 128
   samples: 500
   ```

   And use it with the `--data` argument:

   ```bash
   --data my_config.yaml
   ```
