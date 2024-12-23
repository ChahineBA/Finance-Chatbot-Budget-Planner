from crewai import Task
from agents import information_collector,budget_calculator,savings_tips_provider

# Define Tasks
collection_task = Task(
    description=(
        "1. Parse the dictionary provided by the user ({user_data}).\n"
        "2. Extract the financial data and present it in a clear, structured, and user-friendly format.\n"
        "3. After extracting the financial data, make an interesting observation for the user.\n"
        "4. End with a message saying, 'Please wait for the full analysis.'"
    ),
    expected_output=(
        "A structured and user-friendly display of the financial information from the dictionary, "
        "followed by an interesting observation and a message to wait for the full analysis."
    ),
    agent=information_collector
)

# Define the Budget Calculation Task
budget_task = Task(
    description=(
        "1. Using the provided financial information ({user_data}), calculate a simple budget based on the 50/30/20 rule.\n"
        "2. Allocate 50% for needs (fixed expenses), 30% for wants (discretionary expenses), and 20% for savings.\n"
        "3. If necessary, adjust the distribution based on the user's financial priorities or data.\n"
        "4. Display the budget breakdown in a clear and structured format."
    ),
    expected_output=(
        "A breakdown of the budget based on the 50/30/20 rule, with clear percentages for needs, wants, and savings, "
        "and an option to adjust based on user data or priorities."
    ),
    agent=budget_calculator
)

# Define the Money Saving Tips Task
savings_tips_task = Task(
    description=(
        "1. Analyze the user's financial information ({user_data}) to determine the most effective areas for saving money.\n"
        "2. Offer 1–2 quick and actionable tips to help the user save money. For example:\n"
        "   - 'Try cutting dining expenses by 10% to save more.'\n"
        "   - 'Set up an emergency fund with three months of expenses.'\n"
        "3. Ensure the tips are simple, clear, and relevant to the user's financial situation."
    ),
    expected_output=(
        "1–2 practical and personalized tips for saving money, based on the user's financial data."
    ),
    agent=savings_tips_provider
)