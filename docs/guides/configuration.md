
# GuideLLM Environment Variable Configuration

For advanced users, you can set up a config file that contains environment variables which you can tailor for your guidellm benchmarks reports. These environment variables allow you to control various settings in the application, including logging, dataset preferences, emulated data usage, OpenAI API configuration, and report generation options. Setting these variables in a configuration file or directly in the environment can tailor the application's behavior to different environments or use cases.

## GuideLLM Environment Variable Configuration Details

You can set environment variables directly in your `.env` file or in your environment (e.g., via shell commands) to configure the application as needed. and are split into the following categories: 
1. General Environment Variables 
2. Logging Settings
3. Dataset Settings
4. Emulated Data Settings
5. OpenAI Settings
6. Report Generation Settings 

### 1. **General Environment Variables**

-   **`GUIDELLM__ENV`**:  
    Sets the application's operating environment. It can be one of `local`, `dev`, `staging`, or `prod`. This controls which set of configuration defaults are used (e.g., URLs for reports, log levels, etc.).
    
-   **`GUIDELLM__REQUEST_TIMEOUT`**:  
    Controls the request timeout duration for the application in seconds. The default is 30 seconds.
    
-   **`GUIDELLM__MAX_CONCURRENCY`**:  
    Determines the maximum number of concurrent processes or requests the application can handle. The default is 512.
    
-   **`GUIDELLM__NUM_SWEEP_PROFILES`**:  
    Sets the number of sweep profiles to use. The default is 9.
    

### 2. **Logging Settings**

-   **`GUIDELLM__LOGGING__DISABLED`**:  
    Enables or disables logging for the application. If set to `true`, logging is disabled. Default is `false`.
    
-   **`GUIDELLM__LOGGING__CLEAR_LOGGERS`**:  
    If `true`, existing loggers are cleared when the application starts. Default is `true`.
    
-   **`GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL`**:  
    Sets the logging level for console output (e.g., `INFO`, `WARNING`, `ERROR`). Default is `WARNING`.
    
-   **`GUIDELLM__LOGGING__LOG_FILE`**:  
    Specifies the file path to write logs. If not set, logs are not written to a file.
    
-   **`GUIDELLM__LOGGING__LOG_FILE_LEVEL`**:  
    Sets the logging level for the log file if `LOG_FILE` is specified.
    

### 3. **Dataset Settings**

-   **`GUIDELLM__DATASET__PREFERRED_DATA_COLUMNS`**:  
    A list of preferred column names to use from datasets. This is useful when working with varied datasets that may have different column names for similar data types.
    
-   **`GUIDELLM__DATASET__PREFERRED_DATA_SPLITS`**:  
    A list of preferred dataset splits (e.g., `train`, `test`, `validation`) that the application uses.
    

### 4. **Emulated Data Settings**

-   **`GUIDELLM__EMULATED_DATA__SOURCE`**:  
    URL or path to the source of the emulated data. This is used when running the application with mock data.
    
-   **`GUIDELLM__EMULATED_DATA__FILTER_START`**:  
    A string that marks the start of the text to be used from the `SOURCE`.
    
-   **`GUIDELLM__EMULATED_DATA__FILTER_END`**:  
    A string that marks the end of the text to be used from the `SOURCE`.
    
-   **`GUIDELLM__EMULATED_DATA__CLEAN_TEXT_ARGS`**:  
    A dictionary of boolean settings to control the text cleaning options, such as fixing encoding, removing empty lines, etc.
    

### 5. **OpenAI Settings**

-   **`GUIDELLM__OPENAI__API_KEY`**:  
    The API key for authenticating requests to OpenAI's API.
    
-   **`GUIDELLM__OPENAI__BASE_URL`**:  
    The base URL for the OpenAI server or a compatible server. Default is `http://localhost:8000/v1`.
    
-   **`GUIDELLM__OPENAI__MAX_GEN_TOKENS`**:  
    The maximum number of tokens that can be generated in a single API request. Default is 4096.
    

### 6. **Report Generation Settings**

-   **`GUIDELLM__REPORT_GENERATION__SOURCE`**:  
    The source path or URL from which the report will be generated. If not set, defaults are used based on the environment (`local`, `dev`, `staging`, `prod`).
    
-   **`GUIDELLM__REPORT_GENERATION__REPORT_HTML_MATCH`**:  
    The placeholder string that will be matched in the HTML report to insert data. Default is `"window.report_data = {};"`.
    
-   **`GUIDELLM__REPORT_GENERATION__REPORT_HTML_PLACEHOLDER`**:  
    Placeholder format to be replaced with the generated report data.

## Environment Variable Usage

To use these environment variables in your code for different goals, you can set them in an `.env` file or directly in your environment and then use the `Settings` class from your provided code to access and utilize them. Here’s how you can leverage these settings in your code:

### 1. **Setting Up the Environment Variables**

You can set environment variables directly in your `.env` file or in your environment (e.g., via shell commands) to configure the application as needed. For example:


```bash
# In your .env file or shell
export GUIDELLM__ENV=dev
export GUIDELLM__LOGGING__DISABLED=true
export GUIDELLM__OPENAI__API_KEY=your_openai_api_key
export GUIDELLM__REPORT_GENERATION__SOURCE="https://example.com/report"

```

These settings will be loaded by the `Settings` class from the `.env` file or environment when the application starts.

### 2. **Accessing Environment Variables in Code**

The `Settings` class in the code is powered by `pydantic` and `pydantic_settings`, making it easy to access environment variables in your application code.

For example:
``` python
# Access settings
current_settings = settings

# Print the current environment 
print(f"Current Environment: {current_settings.env}") 

# Check if logging is disabled 
if current_settings.logging.disabled: 
	print("Logging is disabled.") 

# Access OpenAI API key 
openai_api_key = current_settings.openai.api_key 
print(f"Using OpenAI API Key: {openai_api_key}") 

# Generate a report using the source URL 
report_source = current_settings.report_generation.source print(f"Generating report from source: {report_source}")
```

### 3. **Customize Environement Variables for your Goals**

You can utilize the settings for various goals in your code as follows:

#### Goal 1: **Customizing Logging Behavior**

By setting `GUIDELLM__LOGGING__DISABLED`, `GUIDELLM__LOGGING__CLEAR_LOGGERS`, `GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL`, and other logging-related settings, you can control how logging behaves:


```python
if current_settings.logging.disabled:
    # Disable logging in your application
    logger.disabled = True
else:
    # Set logging levels
    logger.setLevel(current_settings.logging.console_log_level)
    # Optionally clear existing loggers
    if current_settings.logging.clear_loggers:
        logging.root.handlers.clear()

    # Log to a file if specified
    if current_settings.logging.log_file:
        file_handler = logging.FileHandler(current_settings.logging.log_file)
        file_handler.setLevel(current_settings.logging.log_file_level or "WARNING")
        logger.addHandler(file_handler)
```


#### Goal 2: **Configuring Dataset Preferences**

If you want to control how your application processes datasets, you can customize dataset-related settings:

``` python
preferred_columns = current_settings.dataset.preferred_data_columns
preferred_splits = current_settings.dataset.preferred_data_splits

# Use preferred columns to filter dataset 
filtered_data = dataset.filter(columns=preferred_columns) 
# Use preferred splits to process only the required data splits 
for split in preferred_splits: 
	process_data_split(data[split])
```

#### Goal 3: **Using Emulated Data for Testing**

To use emulated data for testing, you can adjust the `GUIDELLM__EMULATED_DATA__SOURCE` and related settings:

``` python 
emulated_data_source = current_settings.emulated_data.source
# Read data from the emulated source
with open(emulated_data_source, "r") as f:
    data = f.read()

# Apply filters and cleaning based on settings
filtered_data = apply_filters(data, start=current_settings.emulated_data.filter_start, end=current_settings.emulated_data.filter_end)
cleaned_data = clean_text(filtered_data, **current_settings.emulated_data.clean_text_args)
```

#### Goal 4: **Configuring OpenAI API Requests**
To make API requests to OpenAI or a compatible server, use `GUIDELLM__OPENAI__API_KEY`, `GUIDELLM__OPENAI__BASE_URL`, and other OpenAI-related settings:

``` python 
import requests

headers = {
    "Authorization": f"Bearer {current_settings.openai.api_key}"
}
url = current_settings.openai.base_url

response = requests.post(url, headers=headers, json={
    "prompt": "Translate this to French: 'Hello, world!'",
    "max_tokens": current_settings.openai.max_gen_tokens
})

print(response.json())
```

#### Goal 5: **Generating Reports Dynamically**

You can control report generation behavior based on environment settings:

``` python 
if not current_settings.report_generation.source:
    # Use the default report source based on environment
    report_url = ENV_REPORT_MAPPING[current_settings.env]
else:
    report_url = current_settings.report_generation.source

# Fetch or generate the report
generate_report_from_source(report_url)
```

### 4. **Reloading Settings Dynamically**

To dynamically reload settings based on changes in the environment or `.env` file, you can use the `reload_settings` function:

``` python 
# Reload settings when changes are detected
reload_settings()

# Re-access updated settings
new_settings = settings
print(f"Updated Environment: {new_settings.env}")
```


### 5. **Generating `.env` Files Programmatically**

You can generate a `.env` file programmatically using the `generate_env_file` method:

``` python 
env_file_content = settings.generate_env_file()

with open(".env", "w") as env_file:
    env_file.write(env_file_content)

print("Generated .env file with current settings.")
```

## Conclusion
By configuring and accessing these environment variables in your code, you can effectively manage application settings for various use cases such as development, testing, and production, while dynamically adjusting application behavior without needing to hard-code values directly.
