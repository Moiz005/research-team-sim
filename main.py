import pdfplumber
import redis
from langgraph.graph import StateGraph, END
from typing import Dict, Any
import logging
import os
import json
import arxiv

# Set up logging
logging.basicConfig(filename='reader.log', level=logging.INFO)

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def fetcher_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetcher agent to retrieve paper metadata and PDF from ArXiv API."""
    arxiv_id = state.get("arxiv_id", "1606.00915")  # Default to DeepLab paper
    paper_id = state.get("paper_id", "deeplab_2016")

    try:
        client = arxiv.Client(page_size=1, delay_seconds=3, num_retries=3)
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search), None)
        if not paper:
            logging.error(f"No paper found for ID: {arxiv_id}")
            return {"paper_id": paper_id, "error": f"No paper found for ID: {arxiv_id}"}
        
        metadata = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "doi": paper.doi or "N/A",
            "publication_date": paper.published.strftime("%Y-%m-%d"),
            "abstract": paper.summary
        }

        os.makedirs("papers", exist_ok=True)
        pdf_path = f"papers/{paper_id}.pdf"
        paper.download_pdf(dir_path="./papers", filename=f"{paper_id}.pdf")

        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 1000:
            logging.error(f"Invalid or empty PDF for {arxiv_id}")
            return {"paper_id": paper_id, "error": "Failed to download valid PDF"}
        
        output = {
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "pdf_path": pdf_path,
            "metadata": metadata
        }

        redis_client.set(f"paper/{paper_id}", json.dumps(output))

        logging.info(f"Fetched {arxiv_id}, stored as paper:{paper_id}")

        return {"paper_id": paper_id, "fetcher_output": output}

    except Exception as e:
        logging.error(f"Error fetching {arxiv_id}: {str(e)}")
        return {"paper_id": paper_id, "error": str(e)}