import subprocess
import sys
import time
import csv
import os

# The question to use for all benchmarks (edit as needed or pass via sys.argv)
question = "I'd like to compare hyperscalers to assess which one is the best choice for enterprise use, in about 600 words?"

# Output CSV filenames for each script
azure_csv = "openai_results.csv"
gcp_csv = "vertexai_results.csv"
aws_csv = "bedrock_claude_results.csv"

# Paths to your scripts
azure_script = "azure_openai_demo.py"
gcp_script = "gcp_vertexai_demo.py"
aws_script = "aws_bedrock_claude_demo.py"

# List of scripts to run with their CSV filenames and provider names
scripts = [
    ("Azure OpenAI", azure_script, azure_csv),
    ("GCP Vertex AI", gcp_script, gcp_csv),
    ("AWS Bedrock Claude", aws_script, aws_csv),
]

start_time = time.time()  # Start timing

for name, script, csv_file in scripts:
    print(f"\n=== Running {name} Benchmark ===\n")
    try:
        subprocess.run(
            [sys.executable, script, "--question", question, "--csv", csv_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")

end_time = time.time()  # End timing
elapsed = end_time - start_time

print(f"\nAll benchmarks completed in {elapsed:.2f} seconds.")

# Compile averages from each CSV into a summary file
summary_csv = "benchmark_summary.csv"
summary_rows = []
header = [
    "Provider",
    "Average Response Time (s)",
    "Average Prompt Tokens",
    "Average Completion Tokens",
    "Average Total Tokens",
    "Average Characters",
    "Average Words",
    "Average Cost",
    "Region",
    "Timestamp"
]

for name, _, csv_file in scripts:
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found, skipping.")
        continue
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
        # Find the row that starts with "Average"
        avg_row = next((row for row in rows if row and row[0].strip().lower() == "average"), None)
        if avg_row:
            # Only keep the relevant columns (skip the last column if it's empty or the response)
            summary_rows.append([name] + avg_row[1:8])
        else:
            print(f"Warning: No averages found in {csv_file}")

# Write the summary CSV
with open(summary_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in summary_rows:
        writer.writerow(row)

print(f"\nSummary written to {summary_csv}")

# Transpose the summary so metrics are rows and providers are columns
# First, collect the provider names and their averages (excluding the "Provider" column)
provider_names = [row[0] for row in summary_rows]
averages_by_provider = [row[1:] for row in summary_rows]

# Define the metric names in the order they appear in the averages
metric_names = [
    "Avg. Response Time (s)",
    "Avg. Prompt Tokens",
    "Avg. Completion Tokens",
    "Avg. Total Tokens",
    "Avg. Characters",
    "Avg. Words",
    "Avg. Cost",
    "Region",
    "Timestamp"
]

# Prepare transposed rows: first row is header, then one row per metric
transposed_rows = []
header_row = ["Metric"] + provider_names
transposed_rows.append(header_row)

for i, metric in enumerate(metric_names):
    row = [metric]
    for provider_avg in averages_by_provider:
        # Some scripts may have fewer columns if something failed, so use a default if missing
        value = provider_avg[i] if i < len(provider_avg) else ""
        row.append(value)
    transposed_rows.append(row)

# Write the transposed summary CSV
with open("benchmark_summary_transposed.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in transposed_rows:
        writer.writerow(row)

print("\nTransposed summary written to benchmark_summary_transposed.csv")