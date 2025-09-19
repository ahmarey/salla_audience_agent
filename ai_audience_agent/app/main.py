import os
from typing import List, Literal, Optional, TypedDict, 

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Define the Pydantic Models for Input and Output ---
# This ensures our data is structured and validated.

class ParseRequest(BaseModel):
    """The input request model for the API."""
    prompt: str

class Filter(BaseModel):
    """Represents a single filter condition."""
    field: str = Field(description="The field to filter on. Must be one of the supported fields.")
    operator: str = Field(description="The operator to use for the filter. Must be one of the supported operators.")
    value: str | int | float | list = Field(description="The value for the filter.")

class StructuredOutput(BaseModel):
    """The structured JSON output that the LLM should generate."""
    filters: List[Filter]

# --- 2. Define the LangGraph Agent State ---
# This is the "memory" of our agent as it moves through the graph.

class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        prompt: The initial user prompt.
        filters: The list of parsed filters.
        error: A potential error message.
    """
    prompt: str
    filters: Optional[List[Filter]]
    error: Optional[str]

# --- 3. Define Graph Nodes ---
# These are the functions that will perform actions and modify the state.

def parsing_node(state: AgentState):
    """
    Parses the user prompt into structured filters using an LLM.
    """
    print("--- 🧠 PARSING PROMPT ---")
    
    # This is the master system prompt. It's the most critical part of the logic.
    system_prompt = """
You are an expert at converting natural language queries into structured JSON filters.
Your task is to parse the user's prompt and extract a list of filter conditions.

You must adhere to the following constraints:
1.  The output must be a JSON object that matches this Pydantic schema: `StructuredOutput`.
2.  The "field" must be one of the following supported fields:
    - gender, birthday, birthday_days, joining_date, last_login
    - doesnt_have_orders, have_cancelled_orders, latest_purchase
    - total_sales, total_orders, store_rating
    - doesnt_have_email
    - country, city
3.  The "operator" must be one of the following: =, !=, <, >, <=, >=, between.

Here is an example:
User prompt: "Find customers in Riyadh or Jeddah who joined after Jan 2023 with more than 5 orders."
اعثر على العملاء في الرياض أو جدة الذين انضموا بعد يناير 2023 ولديهم أكثر من 5 طلبات

Your JSON output:
{
    "filters": [
        { "field": "city", "operator": "=", "value": ["Riyadh", "Jeddah"] },
        { "field": "joining_date", "operator": ">", "value": "2023-01-01" },
        { "field": "total_orders", "operator": ">", "value": 5 }
    ]
}
"""
    # Initialize the LLM. We use GPT-4o for its strong instruction-following capabilities.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)

    # Use the .with_structured_output method to guarantee valid JSON
    structured_llm = llm.with_structured_output(StructuredOutput)

    # Invoke the LLM with the system prompt and the user's input
    result = structured_llm.invoke([
        ("system", system_prompt),
        ("human", state["prompt"])
    ])
    
    print(f"--- ✅ PARSING COMPLETE --- \n{result}")

    # Update the state with the parsed filters
    return {"filters": result.filters}


# --- 4. Build the LangGraph ---
# This is where we define the flow of our agent.

# For now, it's a simple, linear graph.
# We will add validation and error handling in the next phase.
workflow = StateGraph(AgentState)
workflow.add_node("parser", parsing_node)

# Set the entry point and the end point
workflow.set_entry_point("parser")
workflow.set_finish_point("parser")

# Compile the graph into a runnable app
app_graph = workflow.compile()


# --- 5. Create the FastAPI Application ---

app = FastAPI(
    title="AI Audience Agent",
    description="An API for parsing natural language prompts into structured audience filters.",
    version="0.1.0",
)

@app.post("/parse_prompt", tags=["Parsing"])
async def parse_prompt_endpoint(request: ParseRequest):
    """
    Receives a natural language prompt and returns structured filters.
    """
    # The input for the graph is a dictionary with the key matching the state.
    graph_input = {"prompt": request.prompt}
    
    # Invoke the graph. The result will be the final state.
    final_state = app_graph.invoke(graph_input)
    
    return {"filters": final_state.get("filters")}

@app.get("/", tags=["Health Check"])
async def root():
    """Root endpoint to check if the API is running."""
    return {"status": "ok", "message": "AI Audience Agent is running!"}