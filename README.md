# Research Team Simulator

A multi-agent system that simulates a research team to read, analyze, and summarize scientific papers using AI agents with distinct roles. Built with Python, LangGraph, and various AI libraries, this project processes papers from arXiv, extracts key concepts, generates summaries, and supports cross-paper insights with dynamic paper discovery.

## Features

**Multi-Agent Architecture**: Comprises Fetcher, Reader, Analyst, Summarizer, and Planner agents:

- **Fetcher**: Downloads papers from arXiv using the `arxiv` library.
- **Reader**: Extracts text and metadata from PDFs using `pdfplumber`.
- **Analyst**: Identifies key concepts and jargon explanations using TF-IDF and `sentence-transformers`.
- **Summarizer**: Generates concise summaries with jargon explanations using an LLM (e.g., `gpt-4o-mini`).
- **Planner**: Decomposes user goals into tasks, dynamically searches for relevant arXiv papers if IDs are not provided, and orchestrates agent execution.

**Dynamic Paper Discovery**: Uses the arXiv API to search for papers based on user goals (e.g., "Survey recent semantic segmentation papers") when arXiv IDs are not specified.

**Long-Term Memory**: Stores processed data (text, keywords, embeddings, summaries) in Redis for persistence and cross-paper analysis.

**Cross-Paper Insights**: Supports similarity searches and clustering on keyword and summary embeddings for comparing papers (e.g., DeepLab and DeepLabv3).

**Optimized Token Usage**: Limits LLM input (e.g., 1500-char text snippets) to reduce costs and improve speed.

**Efficient Task Management**: Uses in-memory task lists to minimize latency, with optional Redis persistence for debugging.

**Error Handling**: Robust logging and validation for reliable processing.

---

## Installation

### Clone the Repository:

```bash
git clone https://github.com/Moiz005/research-team-sim.git
cd research-team-sim

### Install Dependencies:

```bash
pip install arxiv pdfplumber redis langgraph scikit-learn sentence-transformers langchain-openai
```

### Set Up Redis:

* **Install Redis**: Follow instructions for your OS (e.g., `sudo apt install redis-server` on Ubuntu).
* **Start Redis**:

```bash
redis-server
```

### Set OpenAI API Key (for real LLM usage):

```bash
export OPENAI_API_KEY="your_api_key"
```

---

## Usage

### Run the Pipeline

Process papers by specifying a goal and optionally providing arXiv IDs.

```bash
python main.py --goal "Survey recent semantic segmentation papers" --arxiv-ids 1606.00915 1706.05587 --task process
```

If no arXiv IDs are provided, the Planner Agent searches arXiv for relevant papers:

```bash
python main.py --goal "Survey recent semantic segmentation papers" --task process
```

### Perform Similarity Search

Compare papers based on summary embeddings:

```bash
python main.py --goal "Survey recent semantic segmentation papers" --task similarity --query-id paper_1606_00915
```

### Perform Clustering

Cluster papers based on summary embeddings:

```bash
python main.py --goal "Survey recent semantic segmentation papers" --task cluster --query-id paper_1606_00915
```

---

## Verify Output

Check processed data in Redis:

```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
print(json.loads(redis_client.get("paper:paper_1606_00915")))
print(json.loads(redis_client.get("paper:paper_1706_05587")))
```

---

## Cross-Paper Insights

Perform similarity searches programmatically:

```python
import numpy as np
from scipy.spatial.distance import cosine

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
redis_data1 = json.loads(redis_client.get("paper:paper_1606_00915"))
redis_data2 = json.loads(redis_client.get("paper:paper_1706_05587"))

emb1 = np.array(redis_data1["summarizer_output"]["summary_embedding"])
emb2 = np.array(redis_data2["summarizer_output"]["summary_embedding"])

similarity = 1 - cosine(emb1, emb2)
print(f"Similarity between papers: {similarity:.4f}")
```

---

## Project Structure

```
research-team-sim/
├── papers/                            # Directory for downloaded PDFs
├── pipeline.log                       # Log file for processing details
├── main.py                            # Main pipeline script
├── requirements.txt                   # All the requirements for this project
└── README.md                          # This file
```

---

## How It Works

* **Planner Agent**: Decomposes the user goal (e.g., "Survey recent semantic segmentation papers") into tasks, searches arXiv for relevant papers if no IDs are provided, and assigns tasks to other agents.
* **Fetcher Agent**: Downloads arXiv papers and stores metadata in Redis.
* **Reader Agent**: Extracts text and metadata from PDFs, storing results in Redis.
* **Analyst Agent**: Uses TF-IDF to extract keywords (up to 10), generates jargon explanations via LLM, and creates embeddings with sentence-transformers.
* **Summarizer Agent**: Generates 100–200 word summaries using an LLM, incorporating keywords with explanations, and stores summaries with embeddings in Redis.
* **Cross-Paper Insights**: Computes similarity or clusters papers using cosine similarity on embeddings, supporting tasks like comparing DeepLab and DeepLabv3.

---

## Future Improvements

* **Vector Store Integration**: Transition to Chroma or FAISS for scalable similarity searches (10+ papers).
* **Parallel Processing**: Implement concurrent processing for multiple papers using `concurrent.futures` or `asyncio`.
* **Advanced Search**: Enhance arXiv search with category filters (e.g., `cs.CV`) or integrate web search (e.g., Tavily) for broader paper discovery.
* **Task Prioritization**: Prioritize papers based on metadata (e.g., publication date) or relevance scores.
* **Local LLM**: Use a local model (e.g., LLaMA via `vLLM`) to reduce API costs.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature-name
```

3. Commit changes:

```bash
git commit -m "Add feature"
```

4. Push to the branch:

```bash
git push origin feature-name
```

5. Submit a pull request.

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

For questions or suggestions, open an issue or contact [Moiz005](https://github.com/Moiz005).

© 2025 Moiz005

```

Let me know if you want this saved as a downloadable file or adapted into a documentation site (like Docusaurus or MkDocs).
```
