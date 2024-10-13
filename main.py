from agents.main_agent import MainAgent
from agents.travel_planner_agent import TravelPlannerAgent
from tools.feedback_tool import feedback_tool
from tools.travel_planner_tool import TravelPlannerTool
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM and Settings
llm = OpenAI(model="gpt-4o")
Settings.llm = llm

# Initialize tools
travel_planner_tool = TravelPlannerTool()

# Initialize TravelPlannerAgent
travel_planner_agent = TravelPlannerAgent(tools=[travel_planner_tool], llm=llm)

# Initialize main agent worker
main_agent_worker = MainAgent.from_tools(
    tools=[feedback_tool],
    llm=llm,
    verbose=True,
    travel_planner=travel_planner_agent
)

# Create the agent from the worker
main_agent = main_agent_worker.as_agent()

# Start the conversation
task = main_agent.create_task("Start travel planning conversation")
state = main_agent_worker._initialize_state(task)

is_done = False
while not is_done:
    response, is_done = main_agent_worker._run_step(state, task, input=task.input)
    print("TravelBuddy:", response.response)
    if not is_done:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("TravelBuddy: Thank you for using our service. Have a great day!")
            break
        task.input = user_input