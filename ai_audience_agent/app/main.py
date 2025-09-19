import os
from typing import List, Literal, Optional, TypedDict, Any
from datetime import datetime
from dateutil.parser import parse as parse_date # NEW: For parsing dates

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
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
    value: Any = Field(description="The value for the filter.")

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

# --- NEW: 3. Define Supported Filters Schema ---
# This is our "source of truth" for validation.
# We define every allowed field, its operators, and the expected value type.
SUPPORTED_FILTERS = {
    "gender": {"operators": ["=", "!="], "type": str},
    "joining_date": {"operators": ["=", "!=", ">", "<", ">=", "<=", "between"], "type": "date"},
    "total_orders": {"operators": ["=", "!=", ">", "<", ">=", "<="], "type": int},
    "store_rating": {"operators": ["=", "!=", ">", "<", ">=", "<=", "between"], "type": float},
    "city": {"operators": ["=", "!="], "type": (str, list)},
    "country": {"operators": ["=", "!="], "type": (str, list)},
    # Add all other supported fields here...
}


# --- 4. Define Graph Nodes ---

def parsing_node(state: AgentState):
    """Parses the user prompt into structured filters using an LLM."""
    print("\n--- 🧠 PARSING PROMPT ---")
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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)
    structured_llm = llm.with_structured_output(StructuredOutput)
    result = structured_llm.invoke([("system", system_prompt), ("human", state["prompt"])])
    print(f"--- ✅ PARSING COMPLETE ---")
    return {"filters": result.filters}

# NEW: Node to validate the LLM's output
def validation_node(state: AgentState):
    """Validates the parsed filters against the supported schema."""
    print("\n--- 🛡️ VALIDATING FILTERS ---")
    if not state.get("filters"):
        return {"error": "No filters were parsed from the prompt."}

    validated_filters = []
    for f in state["filters"]:
        field_name = f.field
        if field_name not in SUPPORTED_FILTERS:
            error_msg = f"The field '{field_name}' is not supported. Please use one of: {list(SUPPORTED_FILTERS.keys())}"
            print(f"--- ❌ VALIDATION FAILED: {error_msg} ---")
            return {"error": error_msg, "filters": None}
        
        # Add more checks here for operators and value types later...
        
        # Simple date normalization example
        if SUPPORTED_FILTERS[field_name]["type"] == "date":
            try:
                # Attempt to parse the date value
                parsed_value = parse_date(str(f.value))
                f.value = parsed_value.strftime("%Y-%m-%d")
            except ValueError:
                error_msg = f"Invalid date format for field '{field_name}': {f.value}"
                print(f"--- ❌ VALIDATION FAILED: {error_msg} ---")
                return {"error": error_msg, "filters": None}

        validated_filters.append(f)
    
    print("--- ✅ VALIDATION SUCCESSFUL ---")
    return {"filters": validated_filters, "error": None}

# NEW: A simple node to handle errors
def error_node(state: AgentState):
    """A simple node to print out the error and end the graph."""
    print(f"\n--- 🛑 ERROR HANDLED ---")
    print(state.get("error"))
    return {}

# --- NEW: 5. Define Conditional Routing ---
def router(state: AgentState) -> Literal["error_node", "__end__"]:
    """
    This function decides the next step based on the agent's state.
    If an error exists, it routes to the error_node. Otherwise, it ends.
    """
    if state.get("error"):
        return "error_node"
    return "__end__"

# --- 6. Build the LangGraph ---
workflow = StateGraph(AgentState)

workflow.add_node("parser", parsing_node)
workflow.add_node("validator", validation_node)
workflow.add_node("error_handler", error_node)

workflow.set_entry_point("parser")
workflow.add_edge("parser", "validator")
workflow.add_edge("error_handler", END)

# NEW: Add the conditional edge for routing after validation
workflow.add_conditional_edges(
    "validator",
    router,
    {
        "error_node": "error_handler",
        "__end__": END,
    },
)

app_graph = workflow.compile()


# --- 5. Create the FastAPI Application ---

app = FastAPI(
    title="AI Audience Agent",
    description="An API for parsing natural language prompts into structured audience filters.",
    version="0.1.0",
)

@app.post("/parse_prompt", tags=["Parsing"])
async def parse_prompt_endpoint(request: ParseRequest):
    graph_input = {"prompt": request.prompt}
    final_state = app_graph.invoke(graph_input)

    if final_state.get("error"):
        # In a real app, you might want a different status code
        return {"error": final_state["error"]}
        
    return {"filters": final_state.get("filters")}

@app.get("/", tags=["Health Check"])
async def root():
    return {"status": "ok", "message": "AI Audience Agent is running!"}