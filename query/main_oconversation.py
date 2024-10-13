import os
import re
import json
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LlamaIndex settings
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Define slot structure
SLOTS = {
    "duration": None,
    "date": None,
    "group_size": None,
    "occasion": None,
    "trip_type": None,
    "destination": None
}

# Initialize chat memory
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

class TravelAssistant:
    def __init__(self, slot_filling_temp=0.3, itinerary_gen_temp=0.7):
        self.slots = SLOTS.copy()
        self.mandatory_slots = ["duration", "occasion", "destination"]
        self.chat_engine = SimpleChatEngine.from_defaults(memory=memory)
        self.slot_filling_llm = OpenAI(model="gpt-4o", temperature=slot_filling_temp)
        self.itinerary_gen_llm = OpenAI(model="gpt-4o", temperature=itinerary_gen_temp)
        self.itinerary = None
        self.events = []

    def update_slots_from_user_input(self, user_input: str) -> List[str]:
        extraction_prompt = f"""
        Based on the following user input, extract or update the values for these slots:
        {json.dumps(self.slots, indent=2)}

        Mandatory slots: duration, occasion, destination
        If the user explicitly mentions changing or correcting any information, update that slot.
        If a new value is provided for any slot, update it.
        If a slot is not mentioned or no new information is provided, keep its current value.

        User input: {user_input}

        Respond with a pure JSON object containing: Don't include '''json or anything in the beginning.
        1. "slots": The updated slot values
        2. "changed": List of slots that were changed

        For e.g like this 
          
        """

        extraction_response = self.slot_filling_llm.complete(extraction_prompt)
        print(f"\nDEBUG: Raw LLM Response:\n{extraction_response.text}")
        
        try:
            result = json.loads(extraction_response.text)
            extracted_slots = result["slots"]
            changed_slots = result["changed"]
            
            self.slots.update(extracted_slots)
            
            print(f"\nDEBUG: Updated slots:\n{json.dumps(self.slots, indent=2)}")
            print(f"\nDEBUG: Changed slots: {', '.join(changed_slots)}")
            
            return changed_slots
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Problematic response: {extraction_response.text}")
            return []

    def get_missing_mandatory_slots(self) -> List[str]:
        return [slot for slot in self.mandatory_slots if self.slots[slot] is None]

    def chat(self, user_input: str) -> Tuple[str, int]:
        changed_slots = self.update_slots_from_user_input(user_input)
        missing_slots = self.get_missing_mandatory_slots()

        if missing_slots:
            response_prompt = f"""
            The user is planning a trip to {self.slots['destination'] or 'their destination'}. 
            We still need information about: {', '.join(missing_slots)}.
            
            Craft a friendly response that:
            1. Acknowledges any information they've provided
            2. Asks for the missing information in a conversational way
            3. If relevant, explains why this information is important for planning their trip

            User input: {user_input}
            """
            response = self.slot_filling_llm.complete(response_prompt)
            return str(response), 1  # Action code 1: Ask for more information

        else:
            # All mandatory slots are filled, we can now provide meaningful suggestions
            suggestion_prompt = f"""
            The user is planning a {self.slots['occasion']} trip to {self.slots['destination']} for {self.slots['duration']} days.
            
            Based on this information:
            1. Provide 3-5 specific suggestions for activities, places to visit, or experiences that align with the trip's theme.
            2. Ask the user if they would like to hear more about any of these suggestions or if they have any questions.

            Craft a friendly and enthusiastic response incorporating these elements.
            Ensure the response is tailored to the specific trip theme and doesn't repeat previous information.

            User input: {user_input}
            """
            response = self.slot_filling_llm.complete(suggestion_prompt)
            return str(response), 2  # Action code 2: Provide initial suggestions

    def follow_up_chat(self, user_input: str) -> Tuple[str, int]:
        self.suggestion_count = getattr(self, 'suggestion_count', 0)
        
        follow_up_prompt = f"""
        The user has received suggestions for their {self.slots['occasion']} trip to {self.slots['destination']} for {self.slots['duration']} days.

        Previous user input: "{user_input}"
        Number of unprovoked suggestions made so far: {self.suggestion_count}

        Based on the user's response and the current trip context, determine the most appropriate action:

        1. Provide more detailed information about a specific suggestion
        2. Offer additional suggestions (only if explicitly asked or if fewer than 3 unprovoked suggestions have been made)
        3. Transition to generating a detailed itinerary

        Consider the following:
        - If the user expresses ANY desire to move forward with planning, transition to generating a detailed itinerary.
        - Maintain focus on the user's stated preferences and trip type.
        - Avoid repeating information or suggestions the user has already acknowledged.

        Craft a natural and helpful response based on the chosen action.
        If transitioning to generating an itinerary, confirm this with the user.

        Respond with a pure JSON object containing: Don't include '''json or anything in the beginning.
        {{
            "response": "Your crafted response to the user",
            "action_code": 2 or 3,
            "ready_for_itinerary": true/false,
            "explanation": "Brief explanation of the chosen action and readiness assessment"
        }}

        Action code 2 means continue the discussion, while action code 3 means ready to generate itinerary.
        """
        llm_response = self.slot_filling_llm.complete(follow_up_prompt)
        
        try:
            result = json.loads(llm_response.text)
            response = result.get("response", "I'm sorry, I'm having trouble understanding. Could you please rephrase?")
            action_code = result.get("action_code", 2)
            ready_for_itinerary = result.get("ready_for_itinerary", False)
            explanation = result.get("explanation", "No explanation provided")
            
            print(f"\nDEBUG: Action code: {action_code}")
            print(f"\nDEBUG: Ready for itinerary: {ready_for_itinerary}")
            print(f"\nDEBUG: Explanation: {explanation}")
            
            if ready_for_itinerary or "itinerary" in user_input.lower():
                return "Great! I'll generate a detailed itinerary for your trip. One moment please...", 3
            
            return response, action_code
        except json.JSONDecodeError:
            print("Error: Could not parse result from LLM response.")
            return "I apologize, but I'm having trouble processing our conversation. Could you please repeat your last request?", 2

    def generate_itinerary(self) -> str:
        itinerary_prompt = f"""
        Generate a detailed {self.slots['duration']} itinerary for a trip to {self.slots['destination']}.
        The itinerary should focus on the user's preferences and trip type as described below:

        Trip details:
        {json.dumps(self.slots, indent=2)}

        Format the itinerary as follows:
        Day 1:
        - Morning: [Activity] (timing: [start time] - [end time]) - [Brief description]
        - Afternoon: [Activity] (timing: [start time] - [end time]) - [Brief description]
        - Evening: [Activity] (timing: [start time] - [end time]) - [Brief description]

        Day 2:
        ...

        Ensure all activities are consistent with the user's stated preferences and trip type.
        """

        itinerary_response = self.itinerary_gen_llm.complete(itinerary_prompt)
        return itinerary_response.text

    def generate_itinerary(self) -> str:
        itinerary_prompt = f"""
        Based on the following trip details, generate a detailed day-by-day itinerary:
        {json.dumps(self.slots, indent=2)}

        The itinerary should include:
        - Activities and places to visit for each day
        - Suggested timings for each activity
        - Brief descriptions of the attractions

        Format the itinerary as:
        Day 1:
        - Morning: Activity 1
        - Afternoon: Activity 2
        - Evening: Activity 3

        Day 2:
        ...and so on.

        Provide an itinerary for the full duration of the trip.
        """

        itinerary_response = self.itinerary_gen_llm.complete(itinerary_prompt)
        return itinerary_response.text

    def emit_event(self, event_name: str, event_data: Dict):
        self.events.append({"name": event_name, "data": event_data})
        print(f"Event emitted: {event_name}")

class FlightAgent:
    def __init__(self):
        self.llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo")
        self.tools = self.create_tools()
        self.agent = OpenAIAgent.from_tools(self.tools, llm=self.llm, verbose=True)

    def create_tools(self):
        tools = [
            FunctionTool.from_defaults(fn=self.search_flights, name="search_flights"),
            FunctionTool.from_defaults(fn=self.book_flight, name="book_flight"),
        ]
        return tools

    def search_flights(self, origin: str, destination: str, date: str):
        """Search for flights using DuckDuckGo."""
        query = f"flights from {origin} to {destination} on {date}"
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return [result['body'] for result in results]

    def book_flight(self, flight_details: str):
        """Simulate booking a flight."""
        return f"Booking confirmed for flight: {flight_details}"

    def introduce(self):
        return ("Hello! I'm your flight booking assistant. I can help you find and book the perfect flights for your trip. "
                "To get started, could you please tell me your departure city?")

    def process_itinerary_event(self, event):
        itinerary = event['data']['itinerary']
        slots = event['data']['slots']
        
        prompt = f"""
        You are a helpful flight booking assistant. The user has planned a trip with the following details:
        Destination: {slots['destination']}
        Date: {slots['date']}
        Duration: {slots['duration']}
        
        The itinerary is as follows:
        {itinerary}
        
        Please help the user book a suitable flight for this trip. Start by asking for their departure city if it's not provided.
        Then, search for flights and present options to the user. Finally, help them book their chosen flight.
        """
        
        self.agent.chat(prompt)
        return self.introduce()

def main():
    assistant = TravelAssistant()
    print("Welcome to your travel planning assistant! How can I help you plan your trip?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Thank you for using our travel planning service. Have a great trip!")
            break

        response, action_code = assistant.chat(user_input)
        print("Travel Assistant:", response)

        if action_code == 2:  # Continue discussion
            while action_code == 2:
                user_input = input("You: ")
                response, action_code = assistant.follow_up_chat(user_input)
                print("Travel Assistant:", response)

        if action_code == 3:  # Generate itinerary
            print("Travel Assistant: Great! I'll generate a detailed itinerary for your trip. One moment please...")
            itinerary = assistant.generate_itinerary()
            print("Travel Assistant: Here's a suggested itinerary for your trip:")
            print(itinerary)
            print("Is there anything you'd like to change or add to this itinerary?")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()