import streamlit as st
from agents.main_agent import MainAgent
from agents.travel_planner_agent import TravelPlannerAgent
from tools.feedback_tool import feedback_tool
from tools.travel_planner_tool import TravelPlannerTool
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the page configuration
st.set_page_config(page_title="Travel Planner Bot", page_icon="💬")

# Custom CSS to make the header static and hide the Streamlit header/footer
with open('./static/styles.css') as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Initialize the conversation history if not present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Function to initialize the agent (only once)
def initialize_agent():
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

    return main_agent_worker.as_agent(), main_agent_worker

# Initialize agent and state if not already initialized
if "agent" not in st.session_state:
    st.session_state["agent"], st.session_state["agent_worker"] = initialize_agent()

# Function to generate agent response
def get_agent_response(user_input):
    # Retrieve the existing agent and worker from session state
    main_agent = st.session_state["agent"]
    main_agent_worker = st.session_state["agent_worker"]

    # If the conversation state is not initialized, do it once
    if "conversation_state" not in st.session_state:
        task = main_agent.create_task("Start travel planning conversation")
        state = main_agent_worker._initialize_state(task)
        st.session_state["conversation_state"] = state
        st.session_state["task"] = task
    else:
        # Retrieve the state and task from session state
        state = st.session_state["conversation_state"]
        task = st.session_state["task"]

    # Set the user input to the task
    task.input = user_input

    # Run the agent's response generation step
    response, is_done = main_agent_worker._run_step(state, task, input=task.input)

    # Update the conversation state
    st.session_state["conversation_state"] = state

    return response.response

# Display the previous messages
user_avatar = "👩‍💻"
assistant_avatar = "🤖"

st.title('Travel Planner Bot')

for message in st.session_state["messages"]:
    with st.chat_message(
        message["role"],
        avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
    ):
        st.markdown(message["content"])

# Capture new user input
if prompt := st.chat_input("Ask me something about your travel plans..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)

    # Get response from the agent
    response = get_agent_response(prompt)

    # Display assistant response
    with st.chat_message("assistant", avatar=assistant_avatar):
        st.markdown(response)

    # Append assistant response to session state
    st.session_state["messages"].append({"role": "assistant", "content": response})
