# Import necessary modules
from crewai.flow.flow import Flow, listen, start  # For creating and managing the flow of tasks
from crewai import Crew  # For defining and managing agents and tasks
from pydantic import BaseModel  # For creating a structured state model
from typing import Dict  # For type hints
import os  # For managing environment variables
from dotenv import load_dotenv  # For loading environment variables from a .env file
from litellm import completion  # For interacting with the Gemini model
from agents import information_collector, budget_calculator, savings_tips_provider  # Custom agents for tasks
from tasks import collection_task, budget_task, savings_tips_task  # Custom task definitions

# Create Crew instance with defined agents and tasks
tasks = [collection_task, budget_task, savings_tips_task]
agents = [information_collector, budget_calculator, savings_tips_provider]

crew = Crew(
    agents=agents,  # List of agents responsible for handling specific tasks
    tasks=tasks,  # List of tasks to execute
    verbose=True,  # Enables detailed logging for debugging
)

# Set the GEMINI_API_KEY environment variable using the value from the environment or .env file
os.environ['GEMINI_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Define the state to hold user inputs in a dictionary
class UserInputState(BaseModel):
    user_inputs: Dict[str, str] = {}  # Dictionary to store user inputs with corresponding keys

# Define the flow for managing user inputs
class UserInputFlow(Flow[UserInputState]):
    
    # Generate prompts dynamically using the Gemini model
    def get_gemini_prompt(self, previous_inputs: Dict[str, str]):
        # Default prompt for the initial question
        prompt = "Greet the user and ask the user for their monthly income.(Make it sound Like You're an Assistant)"
        
        # Adjust the prompt based on previous inputs
        if 'monthly_income' in previous_inputs:
            prompt = "Now ask the user for their fixed expenses (e.g., rent, bills)."
        
        if 'fixed_expenses' in previous_inputs:
            prompt = "Now ask the user for their discretionary expenses (e.g., dining, hobbies)."
        
        if 'discretionary_expenses' in previous_inputs:
            prompt = "Ask the user for their savings goal (optional)."
        
        # Call the Gemini model to generate a refined prompt
        response = completion(
            model="gemini/gemini-1.5-flash",  # Gemini model version
            messages=[
                {"role": "system", "content": "You are a helpful assistant that guides users through a series of financial questions."},
                {"role": "user", "content": prompt}
            ]
        )
        # Return the generated message content
        return response.choices[0].message.content

    # Get user input based on a dynamically generated prompt
    def get_user_input(self, prompt: str, key: str):
        # Display the assistant's prompt
        print(f"Assistant: {prompt}")
        # Capture user input
        user_input = input("User: ")
        # Store the input in the state
        self.state.user_inputs[key] = user_input
        return user_input

    # First task: Ask for monthly income
    @start()
    def first_task(self):
        # Generate the prompt dynamically
        prompt = self.get_gemini_prompt(self.state.user_inputs)
        # Get the user input for monthly income
        input1 = self.get_user_input(prompt, "monthly_income")
        return f"Monthly income recorded: {input1}"

    # Second task: Ask for fixed expenses
    @listen(first_task)
    def second_task(self, first_result):
        # Generate the prompt dynamically
        prompt = self.get_gemini_prompt(self.state.user_inputs)
        # Get the user input for fixed expenses
        input2 = self.get_user_input(prompt, "fixed_expenses")
        return f"Fixed expenses recorded: {input2}"

    # Third task: Ask for discretionary expenses
    @listen(second_task)
    def third_task(self, second_result):
        # Generate the prompt dynamically
        prompt = self.get_gemini_prompt(self.state.user_inputs)
        # Get the user input for discretionary expenses
        input3 = self.get_user_input(prompt, "discretionary_expenses")
        return f"Discretionary expenses recorded: {input3}"

    # Fourth task: Ask for savings goal (optional)
    @listen(third_task)
    def fourth_task(self, third_result):
        # Generate the prompt dynamically
        prompt = self.get_gemini_prompt(self.state.user_inputs)
        # Get the user input for savings goal
        input4 = self.get_user_input(prompt, "savings_goal")
        return f"Savings goal recorded: {input4}"

# Instantiate and run the flow
flow = UserInputFlow()
result = flow.kickoff()  # Start the flow and capture the final result

# Output the final result and all user inputs
print(f"Final result: {result}")  # Display the final output message
print("All user inputs:", flow.state.user_inputs)  # Display all recorded user inputs

# Execute the Crew tasks with the collected user data
responses = crew.kickoff(inputs={"user_data": flow.state.user_inputs})
for task in tasks:
    # Print the output of each task
    print(task.output)
