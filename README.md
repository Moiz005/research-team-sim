# Research Team Simulator

A multi-agent system that simulates a research team to read, analyze, and summarize scientific papers using AI agents with distinct roles and memory types. Built with Python, LangGraph, and various AI libraries, this project processes papers from arXiv, extracts key concepts, generates summaries, and supports cross-paper insights.

## Features

- **Multi-Agent Architecture**: Comprises Fetcher, Reader, Analyst, and Summarizer agents, each with specific roles:
  - **Fetcher**: Downloads papers from arXiv using the `arxiv` library.
  - **Reader**: Extracts text and metadata from PDFs using `pdfplumber`.
  - **Analyst**: Identifies key concepts and jargon explanations using TF-IDF and `sentence-transformers`.
  - **Summarizer**: Generates concise summaries with jargon explanations using an LLM (e.g., `gpt-4o-mini`).
- **Long-Term Memory**: Stores processed data (text, keywords, embeddings, summaries) in Redis for persistence and cross-paper analysis.
- **Cross-Paper Insights**: Supports similarity searches on keyword and summary embeddings for comparing papers (e.g., *DeepLab* and *DeepLabv3*).
- **Optimized Token Usage**: Limits LLM input (e.g., 1500-char text snippets) to reduce costs and improve speed.
- **Error Handling**: Robust logging and validation for reliable processing.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Moiz005/research-team-sim.git
   cd research-team-sim

2. **Install Dependencies**:

   ```bash
   pip install arxiv pdfplumber redis langgraph scikit-learn sentence-transformers langchain-openai
   ```

3. **Set Up Redis**:

   * **Install Redis**: Follow instructions for your OS (e.g., `sudo apt install redis-server` on Ubuntu).
   * **Start Redis**:

     ```bash
     redis-server
     ```

4. **Set OpenAI API Key** (for real LLM usage):

   ```bash
   export OPENAI_API_KEY="your_api_key"
   ```

## Usage

### Run the Pipeline

Process one or more papers by specifying arXiv IDs.

```bash
python optimized_summarizer_pipeline.py
```

The default script processes DeepLab (arXiv:1606.00915) and DeepLabv3 (arXiv:1706.05587).

### Verify Output

Check processed data in Redis:

```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)
print(json.loads(redis_client.get("paper:paper_1606_00915")))
print(json.loads(redis_client.get("paper:paper_1706_05587")))
```

### Cross-Paper Insights

Perform similarity searches on summaries:

```python
import numpy as np
from scipy.spatial.distance import cosine

redis_data1 = json.loads(redis_client.get("paper:paper_1606_00915"))
redis_data2 = json.loads(redis_client.get("paper:paper_1706_05587"))

emb1 = np.array(redis_data1["summarizer_output"]["summary_embedding"])
emb2 = np.array(redis_data2["summarizer_output"]["summary_embedding"])

similarity = 1 - cosine(emb1, emb2)
print(f"Similarity between papers: {similarity:.4f}")
```

## Project Structure

```
research-team-sim/
├── papers/                            # Directory for downloaded PDFs
├── pipeline.log                       # Log file for processing details
├── optimized_summarizer_pipeline.py  # Main pipeline script
├── requirements.txt                   # All the requirements for this project
└── README.md                          # This file
```

## How It Works

* **Fetcher Agent**: Downloads arXiv papers (e.g., DeepLab) and stores metadata in Redis.
* **Reader Agent**: Extracts text and metadata from PDFs, storing results in Redis.
* **Analyst Agent**: Uses TF-IDF to extract keywords (up to 5), generates jargon explanations via LLM, and creates embeddings with sentence-transformers.
* **Summarizer Agent**: Generates 100-200 word summaries using an LLM (simulated or gpt-4o-mini), incorporating keywords with explanations, and stores summaries with embeddings in Redis.
* **Cross-Paper Insights**: Computes similarity between paper summaries or keywords using cosine similarity on embeddings.

## Future Improvements

* **Vector Store Integration**: Transition to Chroma or FAISS for scalable similarity searches (10+ papers).
* **Parallel Processing**: Implement concurrent processing for multiple papers using `concurrent.futures`.
* **Planner Agent**: Add an agent to orchestrate tasks and prioritize papers based on insights.
* **Advanced Insights**: Support clustering or filtering by metadata (e.g., publication date, authors).
* **Local LLM**: Use a local model (e.g., LLaMA via vLLM) to reduce API costs.

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

## License

MIT License. See `LICENSE` for details.

## Contact

For questions or suggestions, open an issue or contact [Moiz005](https://github.com/Moiz005).

© 2025 Moiz005