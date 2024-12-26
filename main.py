import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict
from crewai.flow.flow import Flow, listen, start
from crewai import Crew
from litellm import completion
from agents import information_collector, budget_calculator, savings_tips_provider
from tasks import collection_task, budget_task, savings_tips_task
import chainlit as cl

# Load environment variables
load_dotenv()

# Set GEMINI_API_KEY
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

# Define tasks and agents
tasks = [collection_task, budget_task, savings_tips_task]
agents = [information_collector, budget_calculator, savings_tips_provider]

# Create a Crew instance
crew = Crew(
    agents=agents,
    tasks=tasks,
    verbose=True,
)

# Define the user input state
class UserInputState(BaseModel):
    user_inputs: Dict[str, str] = {}

# Define the flow
class UserInputFlow(Flow[UserInputState]):

    def get_gemini_prompt(self, previous_inputs: Dict[str, str]):
        prompt = "Greet the user and ask the user for their monthly income. (Make it sound like you're an assistant.)"

        if 'monthly_income' in previous_inputs:
            prompt = "Now ask the user for their fixed expenses (e.g., rent, bills)."

        if 'fixed_expenses' in previous_inputs:
            prompt = "Now ask the user for their discretionary expenses (e.g., dining, hobbies)."

        if 'discretionary_expenses' in previous_inputs:
            prompt = "Ask the user for their savings goal (optional)."

        response = completion(
            model="gemini/gemini-1.5-flash",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that guides users through financial questions."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    async def get_user_input(self, prompt: str, key: str):
        # Use Chainlit's UI to send and receive inputs
        #await cl.Message(content=prompt).send()
        user_input = await cl.AskUserMessage(content=prompt, timeout=30).send()
        self.state.user_inputs[key] = user_input
        return user_input

    @start()
    async def first_task(self):
        prompt = self.get_gemini_prompt(self.state.user_inputs)
        input1 = await self.get_user_input(prompt, "monthly_income")
        return f"Monthly income recorded: {input1}"

    @listen(first_task)
    async def second_task(self, first_result):
        prompt = self.get_gemini_prompt(self.state.user_inputs)
        input2 = await self.get_user_input(prompt, "fixed_expenses")
        return f"Fixed expenses recorded: {input2}"

    @listen(second_task)
    async def third_task(self, second_result):
        prompt = self.get_gemini_prompt(self.state.user_inputs)
        input3 = await self.get_user_input(prompt, "discretionary_expenses")
        return f"Discretionary expenses recorded: {input3}"

    @listen(third_task)
    async def fourth_task(self, third_result):
        prompt = self.get_gemini_prompt(self.state.user_inputs)
        input4 = await self.get_user_input(prompt, "savings_goal")
        return f"Savings goal recorded: {input4}"


@cl.on_chat_start
async def main():
    # Initialize and run the flow
    flow = UserInputFlow()
    flow.kickoff()   
    # Execute Crew tasks with the collected data
    # Send a loading message
    loading_msg = await cl.Message(content="Loading...").send()
    crew.kickoff(inputs={"user_data": flow.state.user_inputs})
    # Remove loading indicator
    await loading_msg.remove()
    for task in tasks:
        await cl.Message(content=f"{task.output}").send()
