import os
import re
from typing import List, Literal, Optional, TypedDict, Any
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

from fastapi import FastAPI, HTTPException
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

# --- 3. Define Supported Filters Schema ---
SUPPORTED_FILTERS = {
    # Customer Attributes
    "gender": {"operators": ["=", "!="], "type": str},
    "birthday": {"operators": ["=", "!=", ">", "<", ">=", "<=", "between"], "type": "date"},
    "birthday_days": {"operators": ["=", "!=", ">", "<", ">=", "<="], "type": int},
    "joining_date": {"operators": ["=", "!=", ">", "<", ">=", "<=", "between"], "type": "date"},
    "last_login": {"operators": ["=", "!=", ">", "<", ">=", "<=", "between"], "type": "date"},
    
    # Behavioral Attributes (Boolean flags)
    "doesnt_have_orders": {"operators": ["="], "type": bool},
    "have_cancelled_orders": {"operators": ["="], "type": bool},
    "latest_purchase": {"operators": ["=", "!=", ">", "<", ">=", "<="], "type": "date"},

    # Sales & Engagement Metrics
    "total_sales": {"operators": ["=", "!=", ">", "<", ">=", "<="], "type": float},
    "total_orders": {"operators": ["=", "!=", ">", "<", ">=", "<="], "type": int},
    "store_rating": {"operators": ["=", "!=", ">", "<", ">=", "<=", "between"], "type": float},

    # Contact Attributes
    "doesnt_have_email": {"operators": ["="], "type": bool},

    # Geographic Attributes
    "country": {"operators": ["=", "!="], "type": (str, list)},
    "city": {"operators": ["=", "!="], "type": (str, list)},
}

def _calculate_delta(num: int, unit: str) -> timedelta:
    """Calculates a timedelta object from a number and a unit string."""
    unit = unit.lower()
    if unit.startswith("day"):
        return timedelta(days=num)
    elif unit.startswith("week"):
        return timedelta(weeks=num)
    elif unit.startswith("month"):
        # Note: This is an approximation
        return timedelta(days=num * 30)
    elif unit.startswith("year"):
        # Note: This doesn't account for leap years
        return timedelta(days=num * 365)
    else:
        # Fails loudly for unsupported units, as you suggested
        raise ValueError(f"Unsupported time unit: '{unit}'")


# Helper function to handle relative dates
def normalize_relative_date(value: str) -> Optional[str]:
    """
    Converts relative date strings into YYYY-MM-DD format.
    """
    text = value.lower().strip()
    today = datetime.now()

    if text == "today":
        return today.strftime("%Y-%m-%d")
    if text == "yesterday":
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # A more robust regex that handles singular and plural units, including "year"
    pattern = r'(\d+)\s+(days?|weeks?|months?|years?)'

    # Handle formats like "7 days ago" or "last 2 years"
    if "ago" in text or "last" in text:
        match = re.search(pattern, text)
        if match:
            num, unit = int(match.group(1)), match.group(2)
            delta = _calculate_delta(num, unit)
            return (today - delta).strftime("%Y-%m-%d")

    # Handle formats like "next 3 weeks"
    if "next" in text:
        match = re.search(pattern, text)
        if match:
            num, unit = int(match.group(1)), match.group(2)
            delta = _calculate_delta(num, unit)
            return (today + delta).strftime("%Y-%m-%d")

    # If no pattern matches, return None so it can be parsed as a static date
    return None

# --- 4. Define Graph Nodes ---

def parsing_node(state: AgentState):
    """Parses the user prompt into structured filters using an LLM."""
    print("\n--- 🧠 PARSING PROMPT ---")
    system_prompt = """
You are an expert at converting natural language queries into structured JSON filters.
Your task is to parse the user's prompt and extract a list of filter conditions.

You must adhere to the following constraints:
1.  The output must be a JSON object that matches this Pydantic schema: `StructuredOutput`.
2.  **CRITICAL RULE: If the user's prompt contains a filter field that is NOT on the supported list, you MUST return an empty list for the "filters" key.** Do not try to guess a similar field.
3.  **DATE RULE: If the user uses a relative date (e.g., "last 30 days", "yesterday", "2 weeks ago"), you MUST convert it to a static string like "last 30 days" or "2 weeks ago".** Do not convert it to a YYYY-MM-DD date yourself.

Supported Fields:
- gender, birthday, birthday_days, joining_date, last_login
- doesnt_have_orders, have_cancelled_orders, latest_purchase
- total_sales, total_orders, store_rating
- doesnt_have_email
- country, city

Supported Operators: =, !=, <, >, <=, >=, between
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

# Node to validate the LLM's output
def validation_node(state: AgentState):
    """Validates, normalizes, and cleans up the parsed filters."""
    print("\n--- 🛡️ VALIDATING FILTERS ---")
    if not state.get("filters"):
        return {"error": "No valid filters were found. The prompt may contain unsupported fields or be too ambiguous."}
    initial_validated_filters = []
    for f in state["filters"]:
        # Check for missing or null values
        if f.value is None:
            error_msg = f"A value is required for the field '{f.field}' but was not found."
            return {"error": error_msg, "filters": None}
            
        field_name = f.field
        # 1. Check if the field is supported
        if field_name not in SUPPORTED_FILTERS:
            error_msg = f"The field '{field_name}' is not supported. Please use one of: {list(SUPPORTED_FILTERS.keys())}"
            return {"error": error_msg, "filters": None}

        field_schema = SUPPORTED_FILTERS[field_name]
        
        # 2. Check if the operator is supported for this field
        if f.operator not in field_schema["operators"]:
            error_msg = f"Operator '{f.operator}' is not supported for field '{field_name}'. Supported operators are: {field_schema['operators']}"
            return {"error": error_msg, "filters": None}

        # 3. Validate and Coerce the value type
        try:
            expected_type = field_schema["type"]
            # Handle multi-value strings like "Riyadh or Jeddah"
            if expected_type == (str, list) and isinstance(f.value, str):
                # Split by comma or "or", trim whitespace, and remove empty strings
                values = [v.strip() for v in re.split(r'\s+or\s+|,', f.value)]
                f.value = [v for v in values if v]

            # Handle 'between' operator specifically
            elif f.operator == "between":
                if isinstance(f.value, str):
                    # Try to split a string like '3,5' into a list
                    values = re.findall(r'[0-9.]+', str(f.value))
                elif isinstance(f.value, list):
                    values = f.value
                else:
                    raise ValueError("must be a list or a parsable string.")
                if len(values) != 2:
                    raise ValueError(f"between operator requires exactly 2 values, but found {len(values)} in '{f.value}'")
                coerced_values = []
                for val in values:
                    if expected_type == int: coerced_values.append(int(float(val)))
                    elif expected_type == float: coerced_values.append(float(val))
                    elif expected_type == "date": coerced_values.append(parse_date(str(val)).strftime("%Y-%m-%d"))
                    else: coerced_values.append(val)
                f.value = coerced_values
            elif expected_type == int: f.value = int(float(f.value))
            elif expected_type == float: f.value = float(f.value)
            elif expected_type == bool:
                if str(f.value).lower() in ["true", "1", "yes"]: f.value = True
                else: f.value = False
            elif expected_type == "date":
                # Check for relative dates first
                normalized_date = normalize_relative_date(str(f.value))
                if normalized_date:
                    if "ago" in str(f.value).lower() or "last" in str(f.value).lower(): f.operator = ">="
                    f.value = normalized_date
                else:
                    f.value = parse_date(str(f.value)).strftime("%Y-%m-%d")

        except (ValueError, TypeError) as e:
            error_msg = f"Invalid value '{f.value}' for field '{field_name}'. Details: {e}"
            return {"error": error_msg, "filters": None}
        initial_validated_filters.append(f)
    
    # 1. Merge filters with the same field and operator
    merged_filters = {}
    for f in initial_validated_filters:
        key = (f.field, f.operator)
        if key not in merged_filters:
            merged_filters[key] = f
        else:
            # If key exists, merge values into a list
            existing_value = merged_filters[key].value
            if not isinstance(existing_value, list):
                existing_value = [existing_value]
            
            new_value = f.value
            if not isinstance(new_value, list):
                new_value = [new_value]
            
            merged_filters[key].value = existing_value + new_value

    final_filters = list(merged_filters.values())

    # 2. Unwrap single-item lists
    for f in final_filters:
        field_schema = SUPPORTED_FILTERS[f.field]
        # Check if the schema supports `str` and the value is a single-item list
        if field_schema["type"] == (str, list) and isinstance(f.value, list) and len(f.value) == 1:
            f.value = f.value[0]

    print("--- ✅ VALIDATION & CLEANUP SUCCESSFUL ---")
    return {"filters": final_filters, "error": None}

def error_node(state: AgentState):
    """A simple node to print out the error and end the graph."""
    print(f"\n--- 🛑 ERROR HANDLED ---")
    print(state.get("error"))
    return {}
# --- 5. Define Conditional Routing ---
def router(state: AgentState) -> Literal["error_node", "__end__"]:
    """This function decides the next step based on the agent's state."""
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

    # Raise a proper HTTP 400 error if validation fails
    if error := final_state.get("error"):
        raise HTTPException(status_code=400, detail=error)
        
    return {"filters": final_state.get("filters")}

@app.get("/", tags=["Health Check"])
async def root():
    return {"status": "ok", "message": "AI Audience Agent is running!"}