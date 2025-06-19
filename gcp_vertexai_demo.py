# gcp_vertexai_demo.py
# This script benchmarks Google Vertex AI Gemini model responses, including timing, token usage, and output statistics.

import os
import time
import csv
import argparse
import datetime
from google import genai
from google.genai import types

# Parse command-line arguments for the question and CSV filename
parser = argparse.ArgumentParser(description="Benchmark GCP Vertex AI Gemini model responses.")
parser.add_argument(
    "--question",
    type=str,
    default="I'd like to compare hyperscalers to assess which one is the best choice for enterprise use, in about 600 words?",
    help="The question to send to the Gemini model."
)
parser.add_argument(
    "--csv",
    type=str,
    default="vertexai_results.csv",
    help="The CSV filename to write results to."
)
args = parser.parse_args()
prompt = args.question
csv_filename = args.csv

def generate():
    # Get the GCP project ID from environment variable
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set.")

    # Set the location as a variable (can be changed as needed)
    location = "global"

    # Initialize the Vertex AI client for Gemini models
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    model = "gemini-2.5-pro"  # Model name to use

    # Use the prompt from the command line
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt)
            ]
        ),
    ]

    # Configure generation parameters and safety settings
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=1,
        seed=0,
        max_output_tokens=3000,  # Allow enough tokens for 600+ words
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            )
        ],
    )

    num_runs = 5  # Number of times to call the API for benchmarking

    # Lists to store metrics for each run
    response_times = []
    prompt_tokens_list = []
    completion_tokens_list = []
    total_tokens_list = []
    responses = []
    timestamps = []  # New list to store timestamps

    # Pricing for Gemini 2.5 Pro (June 2025)
    input_token_price = 0.00125  # USD per 1K input tokens
    output_token_price = 0.01  # USD per 1K output tokens

    costs = []

    # Run the API call multiple times to gather statistics
    for i in range(num_runs):
        start_time = time.time()  # Start timing
        # Generate content using the Gemini model (streaming)
        response_chunks = list(client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ))
        end_time = time.time()  # End timing
        elapsed = end_time - start_time
        response_times.append(elapsed)

        # Concatenate all chunk texts into a single response string
        full_response = "".join(
            chunk.text for chunk in response_chunks
            if hasattr(chunk, "text") and isinstance(chunk.text, str) and chunk.text is not None
        )
        responses.append(full_response)

        # Extract token usage information from the last chunk (if available)
        usage = getattr(response_chunks[-1], "usage_metadata", None) if response_chunks else None
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        total_tokens = getattr(usage, "total_token_count", 0) if usage else 0

        # Store token usage for this run
        prompt_tokens_list.append(prompt_tokens)
        completion_tokens_list.append(completion_tokens)
        total_tokens_list.append(total_tokens)

        # Calculate cost for this run
        input_cost = (prompt_tokens / 1000) * input_token_price
        output_cost = (completion_tokens / 1000) * output_token_price
        total_cost = input_cost + output_cost
        costs.append(total_cost)

        # Count characters and words in the response
        char_count = len(full_response)
        word_count = len(full_response.split())

        # Add timestamp for when the model completes
        completion_time = datetime.datetime.now().isoformat()
        timestamps.append(completion_time)

        # Print metrics for this run
        print(f"Run {i+1}:")
        print(full_response)
        print(f"Response time: {elapsed:.2f} seconds")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Total cost: ${total_cost:.6f}")
        print(f"Characters: {char_count}")
        print(f"Words: {word_count}")
        print(f"Region: {location}")
        print(f"Timestamp: {completion_time}")
        print("-" * 40)

    # Print averages for all runs
    print("Averages over", num_runs, "runs:")
    print(f"Average response time: {sum(response_times)/num_runs:.2f} seconds")
    print(f"Average prompt tokens: {sum(prompt_tokens_list)/num_runs:.2f}")
    print(f"Average completion tokens: {sum(completion_tokens_list)/num_runs:.2f}")
    print(f"Average total tokens: {sum(total_tokens_list)/num_runs:.2f}")
    print(f"Average total cost: ${sum(costs)/num_runs:.6f}")
    print(f"Region: {location}")
    print(f"Timestamp: {completion_time}")

    # Write all results to a CSV file for later analysis
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow([
            "Run", "Response Time (s)", "Prompt Tokens", "Completion Tokens", "Total Tokens",
            "Characters", "Words", "Cost (USD)", "Location", "Timestamp", "Response"
        ])
        # Write each run's data
        for i in range(num_runs):
            resp_text = responses[i]
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
                location,
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
            location,
            timestamps[i],
            ""
        ])

    print(f"Results written to {csv_filename}")

# Run the benchmarking function
if __name__ == "__main__":
    generate()