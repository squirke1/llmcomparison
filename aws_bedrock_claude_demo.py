# Use the native inference API to send a text message to Anthropic Claude.

import boto3
import json
import time
import csv
import argparse
import sys
import datetime

# Parse command-line arguments for the question and CSV filename
parser = argparse.ArgumentParser(description="Benchmark AWS Bedrock Claude model responses.")
parser.add_argument(
    "--question",
    type=str,
    default="I'd like to compare hyperscalers to assess which one is the best choice for enterprise use, in about 600 words?",
    help="The question to send to the Claude model."
)
parser.add_argument(
    "--csv",
    type=str,
    default="bedrock_claude_results.csv",
    help="The CSV filename to write results to."
)
args = parser.parse_args()
prompt = args.question
csv_filename = args.csv

# Set up Bedrock runtime client and model details
region_name = "us-east-1"  # Set region as a variable
client = boto3.client("bedrock-runtime", region_name=region_name)
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
num_runs = 5

# Lists to store metrics for each run
response_times = []
completion_tokens_list = []
prompt_tokens_list = []
total_tokens_list = []
responses = []
costs = []
timestamps = []  # New list to store timestamps

# Pricing for Claude 3 Sonnet (update if you use a different model)
input_token_price = 0.003  # USD per 1K input tokens
output_token_price = 0.015  # USD per 1K output tokens

for i in range(num_runs):
    # Prepare the request payload
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1100,
        "temperature": 1.0,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    # Send the request and measure response time
    try:
        start_time = time.time()
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(native_request)
        )
        end_time = time.time()
        elapsed = end_time - start_time
        response_times.append(elapsed)
    except Exception as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        response_times.append(0)
        responses.append("")
        prompt_tokens_list.append(0)
        completion_tokens_list.append(0)
        total_tokens_list.append(0)
        costs.append(0)
        timestamps.append("")  # Keep lists in sync
        continue

    # Decode the response body
    body_bytes = response["body"].read() if hasattr(response["body"], "read") else response["body"]
    model_response = json.loads(body_bytes.decode("utf-8"))

    # Extract the response text
    try:
        resp_text = model_response["content"][0]["text"]
    except Exception:
        resp_text = ""
    responses.append(resp_text)

    # Extract token usage if available (Bedrock Claude returns usage in 'usage' key)
    usage = model_response.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens

    prompt_tokens_list.append(prompt_tokens)
    completion_tokens_list.append(completion_tokens)
    total_tokens_list.append(total_tokens)

    # Calculate cost for this call
    input_cost = (prompt_tokens / 1000) * input_token_price
    output_cost = (completion_tokens / 1000) * output_token_price
    total_cost = input_cost + output_cost
    costs.append(total_cost)

    # Add timestamp for when the model completes
    completion_time = datetime.datetime.now().isoformat()
    timestamps.append(completion_time)

    # Count characters and words
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
    print(f"Region: {region_name}")
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
            region_name,
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
        region_name,
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

print(sys.prefix != sys.base_prefix)