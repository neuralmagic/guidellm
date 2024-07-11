# guidellm

# Project configuration

The project is configured with environment variables. Check the example in `.env.example`.

```sh
# Create .env file and update the configuration
cp .env.example  .env

# Export all variables
set -o allexport; source .env; set +o allexport
```

## Environment Variables

| Variable        | Default Value         | Description                                                                                                                                                    |
| --------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OPENAI_BASE_URL | http://127.0.0.1:8080 | The host where the `openai` library will make requests to. For running integration tests it is required to have the external OpenAI compatible server running. |
| OPENAI_API_KEY  | invalid               | [OpenAI Platform](https://platform.openai.com/api-keys) to create a new API key. This value is not used for tests.                                             |
