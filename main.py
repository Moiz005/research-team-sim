from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
import pdfplumber
import redis
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
import logging
import os
import json
import arxiv
from typing import TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import re
from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cosine
import argparse

logging.basicConfig(filename='pipeline.log', level=logging.INFO)

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
class AgentState(TypedDict):
    paper_id: str
    arxiv_id: str
    fetcher_output: Dict[str, Any]
    reader_output: Dict[str, Any]
    analyst_output: Dict[str, Any]
    summarizer_output: Dict[str, Any]
    error: str

jargon_glossary = {
    "atrous convolution": "a method using spaced-out filters for larger receptive fields",
    "semantic segmentation": "assigning a class label to each pixel in an image",
    "fully connected crf": "a model using conditional random fields to refine segmentation",
    "deep convolutional nets": "neural networks with multiple convolutional layers for feature extraction"
}

llm = ChatOpenAI(model="gpt-4o-mini")

def llm_jargon_explanation(keyword: str) -> str:
    """Using LLM to generate concise jargon explanations."""
    LLM_PROMPT = f"""
    Given a technical term from an academic paper, provide a concise explanation (20-50 words) suitable for a general audience. Focus on clarity and brevity, avoiding excessive technical detail. Return the explanation only, no additional text.

    Term: {keyword}
    """
    return llm.invoke(LLM_PROMPT)

summarizer_prompt = ChatPromptTemplate([
    SystemMessage(
        content="""Summarize the following academic paper in 100-200 words for a general audience. Include the provided keywords with their explanations in parentheses. Focus on clarity, brevity, and key contributions. Use the paper's metadata for context.

        Return the summary only, no additional text."""
    ),
    HumanMessage(
        content=(
            "Title: {title}\n"
            "Abstract: {abstract}\n"
            "Full Text: {text}\n"
            "Keywords: {keywords}"
        )
    )
])

def llm_summary_generator(text: str, keywords: list, metadata: dict) -> str:
    """Using LLM to generate a concise summary."""
    summarizer_chain = summarizer_prompt | llm
    
    keyword_str = ", ".join([f"{kw['keyword']} ({kw['explanation']})" for kw in keywords])
    text = text[:1500]
    response = summarizer_chain.invoke({
        "title": metadata["title"],
        "abstract": metadata["abstract"],
        "text": text,
        "keywords": keyword_str
    })

    return response.content

def similarity_search(query_embedding: List[float], paper_ids: List[str], embedding_type: str = "summary_embedding") -> List[Dict]:
    """Perform similarity search on embeddings stored in Redis."""
    results = []
    query_vec = np.array(query_embedding)
    
    for paper_id in paper_ids:
        try:
            redis_data = json.loads(redis_client.get(f"paper:{paper_id}") or "{}")
            summarizer_output = redis_data.get("summarizer_output", {})
            target_embedding = summarizer_output.get(embedding_type, None)
            
            if target_embedding:
                target_vec = np.array(target_embedding)
                if len(target_vec) != len(query_vec):
                    logging.warning(f"Invalid embedding dimension for {paper_id}: expected {len(query_vec)}, got {len(target_vec)}")
                    continue
                similarity = 1 - cosine(query_vec, target_vec)  # Cosine similarity
                metadata = redis_data.get("reader_output", {}).get("metadata", {})
                results.append({
                    "paper_id": paper_id,
                    "similarity": similarity,
                    "summary": summarizer_output.get("summary", ""),
                    "title": metadata.get("title", "Unknown"),
                    "authors": metadata.get("authors", ["Unknown"])
                })
            else:
                logging.warning(f"No {embedding_type} found for {paper_id}")
        except Exception as e:
            logging.error(f"Error in similarity search for {paper_id}: {str(e)}")
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:5]  # Return top-5 results

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
        keyword_embeddings = embedder.encode(keywords, show_progress_bar=False).tolist()
        for keyword, embedding in zip(keywords, keyword_embeddings):
            keyword_clean = keyword.lower().strip()
            explanation = llm_jargon_explanation(keyword_clean)
            if len(explanation) > 100 or explanation == "no explanation available":
                explanation = "no clear explanation available"
            enriched_keywords.append({
                "keyword": keyword,
                "explanation": explanation,
                "embedding": embedding
            })

        analyst_output = {
            "paper_id": paper_id,
            "keywords": enriched_keywords
        }

        redis_data = json.loads(redis_client.get(f"paper:{paper_id}"))
        redis_data.update("analyst_output", analyst_output)
        redis_client.set(f"paper:{paper_id}", json.dumps(redis_data))

        logging.info(f"Analyzed {paper_id}, keywords: {len(enriched_keywords)}")

        return {**state, "analyst_output": analyst_output}
    
    except Exception as e:
        logging.error(f"Error analyzing {paper_id}: {str(e)}")
        return {**state, "error": str(e)}

def summarizer_agent(state: AgentState) -> AgentState:
    """Summarizer agent with LLM for dynamic summary generation."""
    reader_output = state['reader_output']
    analyst_output = state['analyst_output']
    paper_id = state["paper_id"]

    if state["error"]:
        logging.error(f"Error Propagated: {state['error']}")
        return {**state, "error": state["error"]}
    
    text = reader_output['text']
    metadata = reader_output['metadata']
    keywords = analyst_output['keywords']

    if not text or not keywords:
        logging.error(f"Missing text or keywords for {paper_id}")
        return {**state, "error": "Missing text or keywords"}
    
    try:
        summary = llm_summary_generator(text, keywords, metadata)

        if len(summary) < 50 or len(summary) > 500:
            logging.warning(f"Invalid summary length for {paper_id}: {len(summary)} chars")
            summary = "Summary generation failed: invalid length"
        
        summary_embedding = embedder.encode([summary], show_progress_bar=False).tolist()[0]

        summarizer_output = {
            "paper_id": paper_id,
            "summary": summary,
            "summary_embedding": summary_embedding
        }

        redis_data = json.loads(redis_client.get(f"paper:{paper_id}") or "{}")
        redis_data.update("summarizer_output", summarizer_output)
        redis_client.set(f"paper:{paper_id}", json.dumps(redis_data))

        logging.info(f"Summarized {paper_id}, summary length: {len(summary)} chars")

        return {**state, "summarizer_output": summarizer_output}
    
    except Exception as e:
        logging.error(f"Error summarizing {paper_id}: {str(e)}")
        return {**state, "error": str(e)}


def process_multiple_papers(arxiv_ids: List[str]) -> List[Dict]:
    """Process multiple papers in sequence."""
    results = []
    for arxiv_id in arxiv_ids:
        paper_id = f"paper_{arxiv_id.replace('.', '_')}"
        initial_state: AgentState = {
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "fetcher_output": {},
            "reader_output": {},
            "analyst_output": {},
            "summarizer_output": {},
            "error": ""
        }
        result = app.invoke(initial_state)
        results.append(result["summarizer_output"]["summary"])
        logging.info(f"Completed processing for arXiv ID: {arxiv_id}")
    return results

graph = StateGraph(AgentState)

graph.add_node("fetcher", fetcher_agent)
graph.add_node("reader", reader_agent)
graph.add_node("analyst", analyst_agent)
graph.add_node("summarizer", summarizer_agent)
graph.set_entry_point("fetcher")
graph.add_edge("fetcher", "reader")
graph.add_edge("reader", "analyst")
graph.add_edge("analyst", "summarizer")
graph.add_edge("summarizer", END)

app = graph.compile()

def main():
    parser = argparse.ArgumentParser(description="Research Team Simulator")
    parser.add_argument("--arxiv-ids", nargs="+", default=["1606.00915"], help="List of arXiv IDs to process")
    parser.add_argument("--task", choices=["process", "similarity"], default="process", help="Task to perform")
    parser.add_argument("--query-id", help="Paper ID for similarity search (e.g., paper_1606_00915)")
    args = parser.parse_args()
    
    if args.task == "process":
        results = process_multiple_papers(args.arxiv_ids)
        print(json.dumps(results, indent=2))
    elif args.task == "similarity":
        if not args.query_id:
            print("Error: --query-id required for similarity task")
            return
        try:
            redis_data = json.loads(redis_client.get(f"paper:{args.query_id}") or "{}")
            query_embedding = redis_data.get("summarizer_output", {}).get("summary_embedding")
            if not query_embedding:
                print(f"Error: No summary embedding found for {args.query_id}")
                return
            results = similarity_search(query_embedding, [f"paper_{aid.replace('.', '_')}" for aid in args.arxiv_ids])
            print("Top similar papers:")
            for paper in results:
                print(f"Paper: {paper['paper_id']}, Title: {paper['title']}, Similarity: {paper['similarity']:.4f}")
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")

if __name__ == "__main__":
    main()