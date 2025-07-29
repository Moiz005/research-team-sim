from langchain_openai import ChatOpenAI
import pdfplumber
import redis
from langgraph.graph import StateGraph, END
from typing import Dict, Any
import logging
import os
import json
import arxiv
from typing import TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import re

logging.basicConfig(filename='reader.log', level=logging.INFO)

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

class AgentState(TypedDict):
    paper_id: str
    arxiv_id: str
    fetcher_output: Dict[str, Any]
    reader_output: Dict[str, Any]
    analyst_output: Dict[str, Any]
    error: str

jargon_glossary = {
    "atrous convolution": "a method using spaced-out filters for larger receptive fields",
    "semantic segmentation": "assigning a class label to each pixel in an image",
    "fully connected crf": "a model using conditional random fields to refine segmentation",
    "deep convolutional nets": "neural networks with multiple convolutional layers for feature extraction"
}

def llm_jargon_explanation(keyword: str) -> str:
    """Using LLM to generate concise jargon explanations."""
    LLM_PROMPT = f"""
    Given a technical term from an academic paper, provide a concise explanation (20-50 words) suitable for a general audience. Focus on clarity and brevity, avoiding excessive technical detail. Return the explanation only, no additional text.

    Term: {keyword}
    """
    llm = ChatOpenAI(model="gpt-4o-mini")
    return llm.invoke(LLM_PROMPT)

def fetcher_agent(state: AgentState) -> AgentState:
    """Fetcher agent to retrieve paper metadata and PDF from ArXiv API."""
    arxiv_id = state['arxiv_id']
    paper_id = state['paper_id']

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
            return {**state, "error": "Failed to download valid PDF"}
        
        fetcher_output = {
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "pdf_path": pdf_path,
            "metadata": metadata
        }

        redis_client.set(f"paper:{paper_id}", json.dumps(fetcher_output))

        logging.info(f"Fetched {arxiv_id}, stored as paper:{paper_id}")

        return {**state, "paper_id": paper_id, "fetcher_output": fetcher_output}

    except Exception as e:
        logging.error(f"Error fetching {arxiv_id}: {str(e)}")
        return {**state, "error": str(e)}

def reader_agent(state: AgentState) -> AgentState:
    """Reader agent to process PDF and extract text/metadata."""
    fetcher_output = state['fetcher_output']
    paper_id = state['paper_id']

    if state['error']:
        logging.error(f"Error in fetcher : {state['error']}")
        return {**state, "error": state['error']}
    
    pdf_path = state['fetcher_output']['pdf_path']
    if not pdf_path or os.path.exists(pdf_path):
        logging.error(f"Invalif pdf path for {paper_id}")
        return {**state, "error": f"Invalif pdf path for {paper_id}"}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
        if len(full_text) < 100:
            logging.warning(f"Poor text extraction for {paper_id}")
            # TODO: Add PyMuPDF or OCR fallback
            return {**state, "error": "Insufficient text extracted"}
        
        reader_output = {
            "paper_id": paper_id,
            "text": full_text,
            "metadata": fetcher_output['metadata']
        }

        redis_data = json.loads(redis_client.get(f"paper:{paper_id}") or "{}")
        redis_data.update({"reader_output": reader_output})
        redis_client.set(f"paper:{paper_id}", json.dumps(redis_data))

        logging.info(f"Processed {paper_id}, text length: {len(full_text)}")

        return {**state, "reader_output": reader_output}
    
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {str(e)}")
        return {**state, "error": str(e)}


def analyst_agent(state: AgentState) -> AgentState:
    """Analyst agent to extract keywords and handle jargon."""
    reader_output = state["reader_output"]
    paper_id = state["paper_id"]
    if state['error']:
        logging.error(f"Error in analyst : {state['error']}")
        return {**state, "error": state['error']}

    text = reader_output['text']
    if not text:
        logging.error(f"No text available for {paper_id}")
        return {**state, "error": "No text available"}
    
    try:
        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out().tolist()

        enriched_keywords = []
        for keyword in keywords:
            keyword_clean = keyword.lower().strip()
            explanation = llm_jargon_explanation(keyword_clean)
            if len(explanation) > 100 or explanation == "no explanation available":
                explanation = "no clear explanation available"
            enriched_keywords.append({
                "keyword": keyword,
                "explanation": explanation
            })

        analyst_output = {
            "paper_id": paper_id,
            "keywords": enriched_keywords,
            "keyword_embeddings": embedder.encode([kw["keyword"] for kw in enriched_keywords]).tolist()
        }

        redis_data = json.loads(redis_client.get(f"paper:{paper_id}"))
        redis_data.update("analyst_output", analyst_output)
        redis_client.set(f"paper:{paper_id}", json.dumps(redis_data))

        logging.info(f"Analyzed {paper_id}, keywords: {len(enriched_keywords)}")

        return {**state, "analyst_output": analyst_output}
    
    except Exception as e:
        logging.error(f"Error analyzing {paper_id}: {str(e)}")
        return {**state, "error": str(e)}

graph = StateGraph(AgentState)

graph.add_node("fetcher", fetcher_agent)
graph.add_node("reader", reader_agent)
graph.add_node("analyst", analyst_agent)
graph.set_entry_point("fetcher")
graph.add_edge("fetcher", "reader")
graph.add_edge("reader", "analyst")
graph.add_edge("analyst", END)

app = graph.compile()

inputs = AgentState(
    paper_id="deeplab_2016",
    arxiv_id="1606.00915",
    fetcher_output={},
    reader_output={},
    analyst_output={},
    error=""
)
response = app.invoke(inputs)

print(json.dumps(response["analust_output"]["keywords"]))