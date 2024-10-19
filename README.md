# AIM Hackathon Oct 2024 - Template
Repository for the first AIM Hackathon together with TIMETOACT GROUP Österreich on 19.10.2024

Challenge explanation and ideas can be found in the [slides](https://timetoactgroup-my.sharepoint.com/:p:/g/personal/felix_krause_timetoact_at/ERZIULziE5pIu7F8MmgPxHwBVuhy9USUtioJsX_Y_31c_g?e=wrWNH4)

<br>

## Set up OpenAI API Key
Copy your teams API key from the [slack](https://join.slack.com/t/aim-ai-impact-mission/shared_invite/zt-2sfahg4h1-Pb7~Ft4ZITZKGAHihEK6QQ) channel description and place it in the `.env_template` file.

You might want to check [OpenAI API pricing](https://openai.com/api/pricing/), to stay within the 20$ limit during the day.

Don't forget to replace the filename to `.env` afterwards!

Check out the sample code to see how to load the key.

<br>

## About the data
The dataset [`reports.json`](data/reports.json) contains the following keys:
- `company_name`: name of the company
- `year`: year of the report (if report is for two years, the first year is used)
- `dataset`: "subdataset" name
    - `handcrafted`: selected by hand, ESG and sustainability reports (23 samples, 2019, 2021, 2023 for every company)
    - `austria`: reports from mainly Austrian companies, hence often in german (33 samples)
    - `scraped`: scraped from a website, might contain broken links and irrelevant pdfs (90 samples, 2016-2024)
- `pdf_url`: link to the pdf report

<br>

## Jump start
### Fork this repository
Simply fork this repository to start working on your project.

### Set up environment
Create a new environment (e.g. with conda)
```bash
conda create -n aim_hackathon_oct24 python=3.10
```

Install the requirements
```bash
pip install -r requirements.txt
```

### Sample code
There is some sample code to help you getting started in [`sample_code.ipynb`](sample_code.ipynb).

<br>

## Hints
Very simple [RAG pipeline](https://medium.com/@ahmed.mohiuddin.architecture/using-ai-to-chat-with-your-documents-leveraging-langchain-faiss-and-openai-3281acfcc4e9) to start with.

You can [extract openAI API token usage](https://help.openai.com/en/articles/6614209-how-do-i-check-my-token-usage) from the response with `response['usage']`.

You can use [tiktoken](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) to manually count tokens of a string:
```bash
import tiktoken
tokenizer = tiktoken.get_encoding("o200k_base")  # for gpt 4o
```

[Structured outputs](https://platform.openai.com/docs/guides/structured-outputs/introduction) force the LLM to output e.g. only integers.

<br>
