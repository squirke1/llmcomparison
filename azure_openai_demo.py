import os
import time
import csv
import argparse
import datetime
from openai import AzureOpenAI

# Parse command-line arguments for the question and CSV filename
parser = argparse.ArgumentParser(description="Benchmark Azure OpenAI model responses.")
parser.add_argument(
    "--question",
    type=str,
    default="I'd like to compare hyperscalers to assess which one is the best choice for enterprise use, in about 600 words?",
    help="The question to send to the Azure OpenAI model."
)
parser.add_argument(
    "--csv",
    type=str,
    default="openai_results.csv",
    help="The CSV filename to write results to."
)
args = parser.parse_args()
prompt = args.question
csv_filename = args.csv

# Load sensitive data from environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # Azure OpenAI endpoint

if endpoint is None:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set.")

model_name = "gpt-4"
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # Deployment name from environment variable

if deployment is None:
    raise ValueError("AZURE_OPENAI_DEPLOYMENT environment variable is not set.")

# Get region/location from environment variable instead of parsing endpoint
region = os.getenv("AZURE_OPENAI_REGION")
if region is None:
    raise ValueError("AZURE_OPENAI_REGION environment variable is not set.")

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint
)

# Pricing for GPT-4.1 (June 2025)
input_token_price = 2.0 / 1000 
output_token_price = 8.0 / 1000

num_runs = 5  # Number of times to call the API for benchmarking

# Lists to store metrics for each run
response_times = []
prompt_tokens_list = []
completion_tokens_list = []
total_tokens_list = []
responses = []
timestamps = []
costs = []  # New list to store cost per run

# Run the API call multiple times to gather statistics
for i in range(num_runs):
    start_time = time.time()
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Hello."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=deployment
    )
    end_time = time.time()
    elapsed = end_time - start_time
    response_times.append(elapsed)

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

    # Calculate cost for this run (divide tokens by 1000)
    input_cost = (prompt_tokens / 1000) * input_token_price
    output_cost = (completion_tokens / 1000) * output_token_price
    total_cost = input_cost + output_cost  # Total cost is the sum of input and output costs

    # Store token usage and cost for this run
    prompt_tokens_list.append(prompt_tokens)
    completion_tokens_list.append(completion_tokens)
    total_tokens_list.append(total_tokens)
    costs.append(total_cost)

    # Get the response text
    resp_text = response.choices[0].message.content or ""
    responses.append(resp_text)

    # Add timestamp for when the model completes
    completion_time = datetime.datetime.now().isoformat()
    timestamps.append(completion_time)

    # Count characters and words in the response
    char_count = len(resp_text)
    word_count = len(resp_text.split())

    # Print metrics for this run
    print(f"Run {i+1}:")
    print(resp_text)
    print(f"Response time: {elapsed:.2f} seconds")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Characters: {char_count}")
    print(f"Words: {word_count}")
    print(f"Cost (USD): {total_cost:.4f}")
    print(f"Region: {region}")
    print(f"Timestamp: {completion_time}")
    print("-" * 40)

# Write all results to a CSV file for later analysis
with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # Write header row
    writer.writerow([
        "Run", "Response Time (s)", "Prompt Tokens", "Completion Tokens", "Total Tokens",
        "Characters", "Words", "Cost (USD)", "Region", "Timestamp", "Response"
    ])
    # Write each run's data
    for i in range(num_runs):
        resp_text = responses[i] or ""
        char_count = len(resp_text)
        word_count = len(resp_text.split())
        writer.writerow([
            i + 1,
            f"{response_times[i]:.2f}",
            prompt_tokens_list[i],
            completion_tokens_list[i],
            total_tokens_list[i],
            char_count,
            word_count,
            f"{costs[i]:.6f}",
            region,
            timestamps[i],
            resp_text.replace('\n', ' ')
        ])
    # Write averages row
    writer.writerow([])
    writer.writerow([
        "Average",
        f"{sum(response_times)/num_runs:.2f}",
        f"{sum(prompt_tokens_list)/num_runs:.2f}",
        f"{sum(completion_tokens_list)/num_runs:.2f}",
        f"{sum(total_tokens_list)/num_runs:.2f}",
        f"{sum(len(r) for r in responses)/num_runs:.2f}",
        f"{sum(len(r.split()) for r in responses)/num_runs:.2f}",
        f"{sum(costs)/num_runs:.6f}",
        region,
        timestamps[i],
        ""
    ])

print(f"Results written to {csv_filename}")

# Print averages to the console for quick reference
print("Averages over", num_runs, "runs:")
print(f"Average response time: {sum(response_times)/num_runs:.2f} seconds")
print(f"Average prompt tokens: {sum(prompt_tokens_list)/num_runs:.2f}")
print(f"Average completion tokens: {sum(completion_tokens_list)/num_runs:.2f}")
print(f"Average total tokens: {sum(total_tokens_list)/num_runs:.2f}")
print(f"Average characters: {sum(len(r) for r in responses)/num_runs:.2f}")
print(f"Average words: {sum(len(r.split()) for r in responses)/num_runs:.2f}")
print(f"Average cost: {sum(costs)/num_runs:.4f} USD")
print(f"Average region: {region}")
print(f"Average timestamp: {timestamps[i]}")