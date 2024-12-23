from crewai import Agent,LLM
import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

my_llm = LLM(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini/gemini-1.5-flash",
)

# Define Agents
information_collector = Agent(
    role="Information Collector",
    goal="Parse the dictionary provided by the user and display the information in a structured and user-friendly format",
    backstory="This agent is skilled at extracting and formatting information from dictionaries, ensuring the output is clear and easy to understand.",
    verbose=True,
    llm=my_llm
)

# Define Agent for Budget Calculation
budget_calculator = Agent(
    role="Budget Calculator",
    goal="Calculate a simple budget using the 50/30/20 rule based on the financial data provided by the user.",
    backstory="This agent specializes in applying the 50/30/20 rule to personal finances. It calculates the distribution of expenses into needs, wants, and savings to help users manage their budget effectively.",
    verbose=True,
    llm=my_llm
)

# Define Agent for Money Saving Tips
savings_tips_provider = Agent(
    role="Savings Tips Provider",
    goal="Offer 1-2 quick tips for saving money based on the user's financial situation.",
    backstory="This agent provides personalized tips for saving money, including actionable suggestions like reducing dining expenses or building an emergency fund.",
    verbose=True,
    llm=my_llm
)