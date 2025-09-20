import json
import pytest
import requests

# URL of the running FastAPI application
API_URL = "http://127.0.0.1:8000/parse_prompt"

# Load the test data from the JSON file
with open("tests/test_data.json", "r") as f:
    test_cases = json.load(f)

# Use pytest.mark.parametrize to create a test for each case in our file
@pytest.mark.parametrize("case", test_cases, ids=[c["id"] for c in test_cases])
def test_audience_agent(case):
    """
    Tests the audience agent API against a dataset of prompts.
    """
    # Send the prompt to the API
    response = requests.post(API_URL, json={"prompt": case["prompt"]})

    # Assert that the HTTP status code is what we expect
    assert response.status_code == case["expected_status"]

    # Assert that the JSON body of the response matches our expected output
    assert response.json() == case["expected_output"]