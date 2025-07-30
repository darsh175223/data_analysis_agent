import os
from dotenv import load_dotenv
import google.generativeai as genai

# get user query -> LLM decides which tools to use -> call tools -> call data analysis tools -> get output -> llm anayzes and smmarrizes results

print("Welcome to the Data Analysis AI Agent!")
query = input("Please enter your data analysis query: ")

tools = []

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_required_tools(query, available_tools):
    prompt = f"""As an expert data analyst, given the following query and list of available tools, 
    precisely identify which tools would be necessary to detect the root cause of any anomaly mentioned in the query.
    
    Query: {query}
    Available Tools: {available_tools if available_tools else 'No tools available'}
    
    If no tools are available, respond with exactly 'No tools'.
    Otherwise, list only the relevant tools needed, with each tool separated by a | symbol."""

    response = model.generate_content(prompt)
    return response.text

# Get required tools based on query
required_tools = get_required_tools(query, tools)
print(f"\nRequired tools for analysis: {required_tools}")

