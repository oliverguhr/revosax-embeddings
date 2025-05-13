import os
import re
import random
import openai
import csv
import sys
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()
markdownFolder = 'data/markdown/'
outputCsv = 'data_import/training-data.csv'
chunk_max_length = 400
chunk_overlap = 20
promptFile = 'data_import/build-training-data-prompt.txt'

## LLM Configuration
base_url = 'https://mms-hackathon-openai.openai.azure.com/'
api_key = os.getenv("API_KEY")
use_azure = True
model = 'gpt-4o-mini'

def split_by_size_with_overlap(chunk):
    """
    Split a chunk by size with overlap, preserving the heading in each sub-chunk.
    """
    # Extract heading if present
    heading_match = re.match(r'(^##\s+.*?)$', chunk, re.MULTILINE)

    if heading_match:
        heading = heading_match.group(0) + '\n'
        # Remove heading from content for splitting
        content = chunk[len(heading):]
    else:
        heading = ""
        content = chunk

    conent_by_words = content.split()
    if (len(conent_by_words) <= chunk_max_length):
        return [chunk]
    
    result = []
    start = 0

    while start < len(conent_by_words):
        # Effective max size accounts for the heading that will be added
        effective_max_size = chunk_max_length

        # Calculate end position
        end = min(start + effective_max_size, len(conent_by_words))

        # Add heading to each sub-chunk
        sub_chunk = [heading] + conent_by_words[start:end]
        result.append(' '.join(sub_chunk))

        # Move start position for next chunk, accounting for overlap
        if end < len(conent_by_words):
            start = end - chunk_overlap
            # Avoid getting stuck if overlap is too large
            if start >= end:
                start = end
        else:
            break

    return result

def load_chunks():   
    chunks = []
    for filename in os.listdir(markdownFolder):
        if filename.endswith('.md'):
            file_path = os.path.join(markdownFolder, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Aufteilen des Inhalts an '##'-Überschriften
            for heading_chunks in re.split(r'(?=^##\s+)', content, flags=re.MULTILINE):
                chunks += split_by_size_with_overlap(heading_chunks)

    return chunks

def init_client():
    if (use_azure):
        client = openai.AzureOpenAI(
                azure_endpoint=base_url,
                api_key=api_key,
                api_version="2024-08-01-preview"
        )
    else:
        client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
        )
    return client

# Function to send a prompt and get a response
def get_completion(client, prompt, chunk):
    """Send a prompt to the OpenAI API and get the response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": chunk}],
            temperature=0.7,  # Controls randomness
            max_tokens=1000    # Limits response length
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    result = []
    chunks = load_chunks()

    sample_size = 0 # 0 = no sample
    if len(sys.argv) > 1:
        sample_size = int(sys.argv[1])

    if sample_size != 0:
        chunks = random.sample(chunks, sample_size)

    client = init_client()
    with open(promptFile, 'r', encoding='utf-8') as f:
        prompt = f.read()

    with open(outputCsv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['chunk', 'result'])  # Kopfzeile

        for chunk in tqdm(chunks):
            current_result = get_completion(client, prompt, chunk)
            writer.writerow([chunk, current_result])
            csvfile.flush()