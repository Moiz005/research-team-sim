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
from sklearn.cluster import KMeans
from pydantic import BaseModel, Field

logging.basicConfig(filename='pipeline.log', level=logging.INFO)

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

class Task(BaseModel):
    task: str = Field(description="Description of the task")
    status: str = Field(description="Task status: pending, in_progress, done, failed")
    assigned_agent: str = Field(description="Agent responsible for the task")
    dependencies: List[str] = Field(description="Tasks that must be completed before this task", default=[])
    output_key: str = Field(description="Key in AgentState to store task output")
    arxiv_id: str = Field(description="arXiv ID associated with the task", default="")

class PlannerState(TypedDict):
    goal: str
    arxiv_ids: List[str]
    tasks: List[Dict]
    completed_tasks: List[Dict]
    current_task: Dict
    error: str

class TaskBreakdown(BaseModel):
    tasks: List[Task] = Field(description="List of tasks to achieve the goal")

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
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:5]

def cluster_papers(paper_ids: List[str], n_clusters: int = 3):
    embeddings = [json.loads(redis_client.get(f"paper:{pid}"))["summarizer_output"]["summary_embedding"] for pid in paper_ids]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(np.array(embeddings))
    return {i: [pid for pid, lbl in zip(paper_ids, labels) if lbl == i] for i in range(n_clusters)}


def search_arxiv(goal: str, max_results: int = 5) -> List[str]:
    """Search arXiv for papers relevant to the goal."""
    try:
        client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=3)
        query = goal.lower().replace("survey", "").strip()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        arxiv_ids = [paper.entry_id.split('/')[-1] for paper in client.results(search)]
        logging.info(f"Found {len(arxiv_ids)} papers for goal: {goal}")
        return arxiv_ids
    except Exception as e:
        logging.error(f"Error searching arXiv for {goal}: {str(e)}")
        return []

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

def planner_agent(state: PlannerState) -> PlannerState:
    """Planner agent to decompose goals into tasks, manage execution with reflexion, and search for arXiv IDs."""
    goal = state['goal']
    
    arxiv_ids = state.get('arxiv_ids', [])
    if not arxiv_ids:
        arxiv_ids = search_arxiv(goal)
        if not arxiv_ids:
            logging.error(f"No arXiv IDs found for goal: {goal}")
            return {**state, "error": f"No relevant papers found for goal: {goal}"}
        state['arxiv_ids'] = arxiv_ids
        redis_client.set(f"planner:search:{goal}", json.dumps({"arxiv_ids": arxiv_ids}))
        logging.info(f"Stored {len(arxiv_ids)} arXiv IDs for goal: {goal}")
    
    tasks = state.get('tasks', [])
    if not tasks:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Planner Agent. Given a user goal related to academic paper analysis and a list of arXiv IDs, break the goal into specific, actionable tasks for each paper. Assign each task to an agent (fetcher, reader, analyst, summarizer) and specify dependencies, output keys, and associate the arXiv ID. Return the tasks in the required format."""),
            HumanMessage(content=f"Goal: {goal}\narXiv IDs: {', '.join(arxiv_ids)}")
        ])
        
        chain = prompt | llm.with_structured_output(TaskBreakdown)
        task_breakdown = chain.invoke({"goal": goal, "arxiv_ids": arxiv_ids})
        
        tasks = [
            {
                "task": task.task,
                "status": task.status,
                "assigned_agent": task.assigned_agent,
                "dependencies": task.dependencies,
                "output_key": task.output_key,
                "arxiv_id": task.arxiv_id
            } for task in task_breakdown.tasks
        ]
        
        redis_client.set(f"planner:global", json.dumps({
            "goal": goal,
            "arxiv_ids": arxiv_ids,
            "tasks": tasks,
            "completed_tasks": [],
            "current_task": {}
        }))
        
        logging.info(f"Decomposed goal '{goal}' into {len(tasks)} tasks for {len(arxiv_ids)} papers")
        return {**state, "arxiv_ids": arxiv_ids, "tasks": tasks, "completed_tasks": [], "current_task": {}}
    
    completed_tasks = state['completed_tasks']
    current_task = state.get('current_task', {})
    
    if current_task and current_task['status'] == "done":
        paper_id = f"paper_{current_task['arxiv_id'].replace('.', '_')}"
        redis_data = json.loads(redis_client.get(f"paper:{paper_id}") or "{}")
        output_key = current_task['output_key']
        output = redis_data.get(output_key, {})
        
        if not output:
            logging.warning(f"Task {current_task['task']} failed validation: no output")
            current_task['status'] = "failed"
            tasks = [t if t['task'] != current_task['task'] else current_task for t in tasks]
            redis_client.set(f"planner:global", json.dumps({
                "goal": goal,
                "arxiv_ids": arxiv_ids,
                "tasks": tasks,
                "completed_tasks": completed_tasks,
                "current_task": {}
            }))
            return {**state, "tasks": tasks, "error": f"Task {current_task['task']} failed validation"}
        
        completed_tasks.append(current_task)
        tasks = [t for t in tasks if t['task'] != current_task['task']]
        current_task = {}
    
    for task in tasks:
        if task['status'] == "pending" and all(dep in [ct['task'] for ct in completed_tasks] for dep in task['dependencies']):
            task['status'] = "in_progress"
            current_task = task
            break
    
    if not current_task and not tasks:
        logging.info(f"All tasks completed for goal: {goal}")
        return {**state, "tasks": [], "completed_tasks": completed_tasks, "current_task": {}, "error": ""}
    
    redis_client.set(f"planner:global", json.dumps({
        "goal": goal,
        "arxiv_ids": arxiv_ids,
        "tasks": tasks,
        "completed_tasks": completed_tasks,
        "current_task": current_task
    }))
    
    logging.info(f"Planner assigned task: {current_task.get('task', 'None')}")
    return {**state, "tasks": tasks, "completed_tasks": completed_tasks, "current_task": current_task}


def router(state: PlannerState) -> str:
    """Router to determine next node based on current task."""
    current_task = state.get('current_task', {})
    if not current_task:
        return END
    agent = current_task.get('assigned_agent', '')
    if agent in ['fetcher', 'reader', 'analyst', 'summarizer']:
        return agent
    return END

def process_multiple_papers(goal: str, arxiv_ids: List[str] = None) -> List[Dict]:
    """Process papers with Planner Agent, using provided or searched arXiv IDs."""
    results = []
    arxiv_ids = arxiv_ids or []
    
    # Initialize PlannerState to get arXiv IDs and tasks
    planner_state: PlannerState = {
        "goal": goal,
        "arxiv_ids": arxiv_ids,
        "tasks": [],
        "completed_tasks": [],
        "current_task": {},
        "error": ""
    }
    
    # Run planner to generate arXiv IDs and tasks
    planner_state = planner_agent(planner_state)
    if planner_state['error']:
        logging.error(f"Failed to process papers: {planner_state['error']}")
        return [{"error": planner_state['error']}]
    
    arxiv_ids = planner_state['arxiv_ids']
    all_tasks = planner_state['tasks']  # Store full task list

    for arxiv_id in arxiv_ids:
        paper_id = f"paper_{arxiv_id.replace('.', '_')}"
        agent_state: AgentState = {
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "fetcher_output": {},
            "reader_output": {},
            "analyst_output": {},
            "summarizer_output": {},
            "error": ""
        }
        
        # Create a local planner state for this paper
        local_planner_state: PlannerState = {
            "goal": goal,
            "arxiv_ids": [arxiv_id],
            "tasks": [t for t in all_tasks if t['arxiv_id'] == arxiv_id],
            "completed_tasks": [],
            "current_task": {},
            "error": ""
        }
        
        # Process tasks for this paper
        while local_planner_state['tasks'] or local_planner_state['current_task']:
            local_planner_state = planner_agent(local_planner_state)
            if local_planner_state['error']:
                results.append({"paper_id": paper_id, "error": local_planner_state['error']})
                break
            
            if local_planner_state['current_task']:
                agent = local_planner_state['current_task']['assigned_agent']
                if agent == 'fetcher':
                    agent_state = fetcher_agent(agent_state)
                elif agent == 'reader':
                    agent_state = reader_agent(agent_state)
                elif agent == 'analyst':
                    agent_state = analyst_agent(agent_state)
                elif agent == 'summarizer':
                    agent_state = summarizer_agent(agent_state)
                
                if agent_state['error']:
                    local_planner_state['error'] = agent_state['error']
                    results.append({"paper_id": paper_id, "error": agent_state['error']})
                    break
                
                # Update task status in all_tasks
                current_task = local_planner_state['current_task']
                current_task['status'] = "done"
                all_tasks = [t if t['task'] != current_task['task'] else current_task for t in all_tasks]
                local_planner_state['completed_tasks'].append(current_task)
                local_planner_state['tasks'] = [t for t in local_planner_state['tasks'] if t['task'] != current_task['task']]
                local_planner_state['current_task'] = {}
        
        if not local_planner_state['error']:
            results.append(agent_state["summarizer_output"])
            logging.info(f"Completed processing for arXiv ID: {arxiv_id}")
    
    # Optionally save final state to Redis for persistence
    redis_client.set(f"planner:global", json.dumps({
        "goal": goal,
        "arxiv_ids": arxiv_ids,
        "tasks": [t for t in all_tasks if t['status'] != "done"],
        "completed_tasks": [t for t in all_tasks if t['status'] == "done"],
        "current_task": {}
    }))
    
    return results

graph = StateGraph(AgentState)

graph.add_node("planner", planner_agent)
graph.add_node("fetcher", fetcher_agent)
graph.add_node("reader", reader_agent)
graph.add_node("analyst", analyst_agent)
graph.add_node("summarizer", summarizer_agent)
graph.set_entry_point("planner")
graph.add_conditional_edges("planner", router, {
    "fetcher": "fetcher",
    "reader": "reader",
    "analyst": "analyst",
    "summarizer": "summarizer",
    END: END
})
graph.add_edge("fetcher", "planner")
graph.add_edge("reader", "planner")
graph.add_edge("analyst", "planner")

app = graph.compile()

def main():
    parser = argparse.ArgumentParser(description="Research Team Simulator with Planner")
    parser.add_argument("--goal", default="Semantic segmentation papers", help="Goal for the planner")
    parser.add_argument("--arxiv-ids", nargs="*", default=[], help="Optional list of arXiv IDs to process")
    parser.add_argument("--task", choices=["process", "similarity", "cluster"], default="process", help="Task to perform")
    parser.add_argument("--query-id", help="Paper ID for similarity search (e.g., paper_1606_00915)")
    args = parser.parse_args()
    
    if args.task == "process":
        results = process_multiple_papers(args.goal, args.arxiv_ids)
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
            paper_ids = [f"paper_{aid.replace('.', '_')}" for aid in args.arxiv_ids] if args.arxiv_ids else [k.split(':')[1] for k in redis_client.keys("paper:*")]
            results = similarity_search(query_embedding, paper_ids)
            print("Top similar papers:")
            for paper in results:
                print(f"Paper: {paper['paper_id']}, Title: {paper['title']}, Similarity: {paper['similarity']:.4f}")
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
    elif args.task == "cluster":
        if not args.query_id:
            print("Error: --query-id required for cluster task")
            return
        try:
            print("Enter cluster size:")
            num_clusters = int(input())
            paper_ids = [f"paper_{aid.replace('.', '_')}" for aid in args.arxiv_ids] if args.arxiv_ids else [k.split(':')[1] for k in redis_client.keys("paper:*")]
            results = cluster_papers(paper_ids, num_clusters)
            print(json.dumps(results, indent=2))
        except Exception as e:
            print(f"Error in clustering: {str(e)}")

if __name__ == "__main__":
    main()