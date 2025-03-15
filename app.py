from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
import os
import uvicorn
import re
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import asyncio

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Deep Research Functions
def split_query_into_subqueries(query: str) -> list:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    prompt_template = (
        "Break down the following complex query into a list of detailed sub-queries that can be individually researched. Maximum of 8 subqueries.\n\n"
        "Query: {query}\n\n"
        "Sub-queries (one per line):"
    )
    prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(query=query)
    subqueries = [line.strip("- ").strip() for line in response.splitlines() if line.strip()]
    return subqueries

def search_subquery(query: str):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You are a helpful research assistant. You provide very specific and concise answers."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": query
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "text"
            }
        },
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {
                    "type": "approximate"
                },
                "search_context_size": "medium"
            }
        ],
        temperature=0.25,
        max_output_tokens=2500,
        top_p=1,
        store=True
    )
    
    answer = response.output_text
    sources = []
    links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', answer)
    for title, url in links:
        if 'utm_source=openai' in url:
            url = url.replace('?utm_source=openai', '')
        sources.append({"title": title, "url": url})
    return answer, sources

def synthesize_final_answer(original_query: str, aggregated_research: str) -> str:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    prompt_template = (
        "Based on the aggregated research below, provide a comprehensive answer to the original query. In your response, use markdown formatting. Do not include hyperlinks or links of any kind.\n\n"
        "Aggregated Research:\n{research}\n\n"
        "Original Query: {query}\n\n"
        "Answer:"
    )
    prompt = PromptTemplate(input_variables=["research", "query"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    final_answer = chain.run(research=aggregated_research, query=original_query)
    return final_answer

async def deep_research_pipeline(query: str):
    # 1. Split query into subqueries
    subqueries = split_query_into_subqueries(query)
    yield {"type": "subqueries", "count": len(subqueries)}
    
    aggregated_research = ""
    aggregated_sources = []
    
    # 2. Research each subquery
    for idx, subquery in enumerate(subqueries):
        yield {"type": "progress", "subquery": subquery, "index": idx + 1, "total": len(subqueries)}
        answer, sources = search_subquery(subquery)
        aggregated_research += f"\n---\nSubquery: {subquery}\nResult:\n{answer}\n"
        aggregated_sources.extend(sources)

    # Remove duplicates and sort sources
    unique_sources = []
    for source in aggregated_sources:
        if source not in unique_sources:
            unique_sources.append(source)
    unique_sources.sort(key=lambda x: x['title'])
    
    # 3. Synthesize final answer
    final_answer = synthesize_final_answer(query, aggregated_research)
    yield {"type": "complete", "answer": final_answer, "sources": unique_sources}

async def stream_deep_research(query: str):
    async for result in deep_research_pipeline(query):
        yield f"data: {json.dumps(result)}\n\n"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str, deep_research: bool = False):
    return templates.TemplateResponse(
        "answer.html", 
        {
            "request": request,
            "query": q,
            "answer": "",
            "sources": [],
            "num_sources": 0,
            "is_loading": True,
            "deep_research": deep_research
        }
    )

@app.get("/api/search")
async def api_search(q: str, deep_research: bool = False):
    if not deep_research:
        # Regular search
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a helpful research assistant."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": q
                        }
                    ]
                }
            ],
            text={
                "format": {
                    "type": "text"
                }
            },
            reasoning={},
            tools=[
                {
                    "type": "web_search_preview",
                    "user_location": {
                        "type": "approximate"
                    },
                    "search_context_size": "medium"
                }
            ],
            temperature=0.25,
            max_output_tokens=2500,
            top_p=1,
            store=True
        )
        
        answer = response.output_text
        sources = []
        
        links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', answer)
        for title, url in links:
            if 'utm_source=openai' in url:
                url = url.replace('?utm_source=openai', '')
            sources.append({"title": title, "url": url})
        
        clean_answer = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\1', answer)
        
        return JSONResponse({
            "answer": clean_answer,
            "sources": sources,
            "num_sources": len(sources)
        })
    else:
        # Deep research - return a streaming response
        return StreamingResponse(
            stream_deep_research(q),
            media_type="text/event-stream"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
