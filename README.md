# Hyperscalers Benchmarking

This repository contains a practical benchmarking toolkit for comparing the performance and output of large language models (LLMs) from the three major cloud providers: **Azure OpenAI GPT-4.1**, **Google Vertex AI Gemini**, and **AWS Bedrock Claude**. The suite is designed to help developers, architects, and decision-makers evaluate these services side by side using real API calls and consistent prompts.

Please see my [Substack post](https://stephenquirke.substack.com/) detailing the project.
---

## What these scripts do

- **Runs the same prompt** across Azure, Google, and AWS LLM APIs.
- **Measures and records** response time, token usage, character count, and word count.
- **Exports results** and averages to CSV files for each provider.
- **Aggregates and summarizes** results for easy comparison.
- **Lets you customize** the prompt and output file via command-line arguments.

---

## Requirements

- Python 3.8 or newer
- API access and credentials for:
    - Azure OpenAI (GPT-4.1 deployment)
    - Google Vertex AI (Gemini model)
    - AWS Bedrock (Claude model)
- Python packages: `boto3`, `google-genai`, `openai`

---

## Setup

### 1. Clone the Repository

```sh
git clone https://github.com/squirke1/hyperscalers
cd hyperscalers
```

### 2. Set Environment Variables

#### Azure OpenAI
```sh
export AZURE_OPENAI_API_KEY="azure-openai-api-key"
export AZURE_OPENAI_ENDPOINT="https://<endpoint>.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="deployment-name"
```

#### **Google Vertex AI**
```sh
export GOOGLE_CLOUD_PROJECT="gcp-project-id"
# Make sure you are authenticated with Google Cloud SDK and have access to Vertex AI
```

#### **AWS Bedrock**
```sh
export AWS_ACCESS_KEY_ID="aws-access-key"
export AWS_SECRET_ACCESS_KEY="aws-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
# Ensure your IAM user/role has Bedrock invoke permissions and model access
```

### 3. Install Python Dependencies

```sh
pip install boto3 google-genai openai
```

---

## Usage

### Run All Benchmarks

```sh
python run_all_benchmarks.py
```

- By default, this will use the standard comparison prompt.
- To use a custom question, edit the `question` variable in `run_all_benchmarks.py` or modify the script to accept a command-line argument.

### Run a Single Provider

Each provider script can be run individually:

```sh
python azure_openai_demo.py --question "custom question" --csv "azure_results.csv"
python gcp_vertexai_demo.py --question "custom question" --csv "gcp_results.csv"
python aws_bedrock_claude_demo.py --question "custom question" --csv "aws_results.csv"
```

---

## Output

- Each script writes detailed results and averages to its own CSV file (e.g., `openai_results.csv`, `vertexai_results.csv`, `bedrock_claude_results.csv`).
- After all scripts run, `run_all_benchmarks.py` creates:
  - `benchmark_summary.csv` — Averages from each provider (providers as rows).
  - `benchmark_summary_transposed.csv` — Averages from each provider (metrics as rows, providers as columns) for easy comparison.

---

## Example Summary Table

| Metric                   | Azure OpenAI GPT-4.1 | Vertex AI Gemini 1.5 Pro | AWS Bedrock Claude 3 Haiku |
|--------------------------|----------------------|--------------------------|----------------------------|
| Avg. Response Time (s)   | 9.61                 | 18.7                     | 9.73                       |
| Avg. Prompt Tokens       | 38                   | 29                       | 35                         |
| Avg. Completion Tokens   | 942.6                | 1226.2                   | 793.6                      |
| Avg. Total Tokens        | 980.6                | 1255.2                   | 828.6                      |
| Avg. Word Count          | 642.4                | 851.4                    | 581.2                      |

---

## Notes

- Ensure you have access to the required models and regions in each cloud provider.
- API usage may incur costs—monitor your usage in each provider's console.
- For best results, use the same or similar prompt for all providers.

---

## License

MIT License

---

## Contact

For questions or contributions, please open an issue or submit a pull request.
