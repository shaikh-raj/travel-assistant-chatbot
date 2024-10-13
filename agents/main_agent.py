import ast
from llama_index.core.agent import CustomSimpleAgentWorker, Task, AgentChatResponse
from llama_index.core.tools import BaseTool
from typing import Dict, Any, List, Tuple, Optional
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from agents.travel_planner_agent import TravelPlannerAgent
import json

class MainAgent(CustomSimpleAgentWorker):
    """Agent worker for travel planning."""

    def __init__(self, tools: List[BaseTool], travel_planner: TravelPlannerAgent, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(tools=tools, **kwargs)
        self._travel_planner = travel_planner
        print("MainAgent initialized with TravelPlannerAgent")

    def _initialize_state(self, task: Task, **kwargs: Any) -> Dict[str, Any]:
        """Initialize state."""
        return {"preferences": {}, "current_step": "initial", "travel_info": None, "itinerary": None}

    def _run_step(
        self, state: Dict[str, Any], task: Task, input: Optional[str] = None
    ) -> Tuple[AgentChatResponse, bool]:
        """Run step."""
        current_step = state.get("current_step", "initial")
        
        print(f"\n--- Current State ---\nStep: {current_step}\nPreferences: {json.dumps(state.get('preferences', {}), indent=2)}\nInput: {input}\n")
        
        if current_step == "initial":
            message = ChatMessage(
                role=MessageRole.USER,
                content="Generate an enthusiastic welcome message for a travel planning assistant named TravelBuddy"
            )
            response = self.llm.chat([message])
            state["current_step"] = "collect_preferences"
            return AgentChatResponse(response=response.message.content), False

        elif current_step == "collect_preferences":
            response, is_done = self._collect_preferences(state, input)
            if is_done:
                state["current_step"] = "generate_itinerary"
                # Generate the itinerary without waiting for user input
                itinerary, session_id = self._travel_planner.generate_itinerary(state["preferences"])
                state["itinerary"] = itinerary
                state["session_id"] = session_id
                response = self._present_itinerary(state)
                state["current_step"] = "get_itinerary_feedback"
            return response, False

        elif current_step == "generate_itinerary":
            # This block should now only be reached if there's an unexpected state
            itinerary, session_id = self._travel_planner.generate_itinerary(state["preferences"])
            state["itinerary"] = itinerary
            state["session_id"] = session_id
            response = self._present_itinerary(state)
            state["current_step"] = "get_itinerary_feedback"
            return response, False

        elif current_step == "get_itinerary_feedback":
            response, is_done = self._handle_itinerary_feedback(state, input)
            return response, is_done

        elif current_step == "offer_booking":
            response, is_done = self._offer_booking(state, input)
            return response, is_done

        return AgentChatResponse(response="I'm not sure how to proceed."), True

    def _collect_preferences(self, state: Dict[str, Any], input: Optional[str]) -> Tuple[AgentChatResponse, bool]:
        preferences = state.get("preferences", {})
        
        if input is not None:
            feedback_tool = next(tool for tool in self.tools if tool.metadata.name == "feedback_processor")
            result = feedback_tool(feedback=input, current_preferences=json.dumps(preferences))
            print(f"\n--- Feedback Tool Output ---\n{result}\n")
            
            if not result.is_error:
                # Use ast.literal_eval to parse the content
                result_dict = ast.literal_eval(result.content)
                if result_dict["status"] == "success":
                    state["preferences"].update(result_dict["updated_preferences"])
                    print(f"\n--- Updated Preferences ---\n{json.dumps(state['preferences'], indent=2)}\n")
                else:
                    print(f"Error in processing feedback: {result_dict['message']}")
            else:
                print(f"Error in feedback tool: {result.raw_output}")

        if state["preferences"].get("need_extra_suggestions", True):  # Assuming we need at least 4 preferences
            next_question = self._get_next_question(state["preferences"])
            return AgentChatResponse(response=next_question), False
        else:
            return AgentChatResponse(response=f"Great! I'm excited to plan your trip to {state['preferences'].get('destination', 'your chosen destination')}. Let me generate an itinerary for you."), True

    def _get_next_question(self, preferences: Dict[str, Any]) -> str:
        need_extra_suggestions = preferences.get("need_extra_suggestions", True)
        suggestion_count = preferences.get("suggestion_count", 0)
        
        preference_fields = [
            'destination', 'start_date', 'end_date', 'budget', 
            'accommodation_type', 'activities', 'transportation'
        ]
        missing_preferences = [pref for pref in preference_fields if pref not in preferences]

        if not missing_preferences and need_extra_suggestions and suggestion_count < 3:
            # Generate a trip suggestion
            suggestion = self._generate_trip_suggestion(preferences)
            preferences["suggestion_count"] = suggestion_count + 1
            return suggestion
        elif not missing_preferences or not need_extra_suggestions:
            preferences["suggestion_count"] = 0
            return "Great! I have all the information I need. Let me prepare an itinerary for you based on your preferences."

        prompt = f"""
        As an enthusiastic travel planning assistant named TravelBuddy, generate a natural and engaging question to ask the user about their travel preferences. 
        
        Current known preferences: {json.dumps(preferences, indent=2)}
        
        Missing preferences: {', '.join(missing_preferences)}
        
        Guidelines:
        1. Be enthusiastic and friendly in your tone.
        2. If this is the first question, ask about multiple preferences (e.g., destination, dates, and budget).
        3. For follow-up questions, focus on 1-2 missing preferences, but phrase it naturally.
        4. Use emojis occasionally to add a fun element.
        5. Reference known preferences to make the question contextual.
        6. Vary your phrasing to keep the conversation interesting.
        7. Occasionally ask about preferences not in the main list, like favorite cuisines or travel style.

        Generate a single question following these guidelines.
        """
        question_llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
        response = question_llm.complete(prompt)
        return response.text.strip()

    def _generate_trip_suggestion(self, preferences: Dict[str, Any]) -> str:
        prompt = f"""
        As TravelBuddy, generate an exciting and personalized trip suggestion based on these preferences:

        {json.dumps(preferences, indent=2)}

        The suggestion should:
        1. Recommend a specific activity, attraction, or experience.
        2. Explain why it's a great fit for their trip.
        3. Ask if they want to include it in their itinerary.
        4. Be enthusiastic and use an emoji or two for fun.

        Provide a single, engaging suggestion.
        """
        suggestion_llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
        response = suggestion_llm.complete(prompt)
        return response.text.strip()

    def _present_itinerary(self, state: Dict[str, Any]) -> AgentChatResponse:
        itinerary = state["itinerary"]
        return AgentChatResponse(response=f"Here's the itinerary I've prepared for you:\n\n{itinerary['full_itinerary']}\n\nDo you like this itinerary?")

    def _handle_itinerary_feedback(self, state: Dict[str, Any], feedback: str) -> Tuple[AgentChatResponse, bool]:
        if 'yes' in feedback.lower():
            state["current_step"] = "offer_booking"
            return AgentChatResponse(response="That's great! I'm glad you like the itinerary. Would you like to book flights or accommodation for this trip?"), False
        else:
            feedback_tool = next(tool for tool in self.tools if tool.metadata.name == "feedback_processor")
            result = feedback_tool(feedback=feedback, current_preferences=json.dumps(state["preferences"]))
            result_dict = ast.literal_eval(result.content)
            
            if result_dict["status"] == "success":
                state["preferences"].update(result_dict["updated_preferences"])
                state["current_step"] = "generate_itinerary"
                state["itinerary"] = None
                return AgentChatResponse(response=f"I understand. Let me adjust the itinerary based on your feedback: {result_dict['change_summary']}. I'll generate a new itinerary that better matches your preferences."), False
            else:
                return AgentChatResponse(response=f"I'm sorry, but there was an error processing your feedback. {result_dict['message']}"), True

    def _offer_booking(self, state: Dict[str, Any], input: Optional[str]) -> Tuple[AgentChatResponse, bool]:
        if input is None:
            return AgentChatResponse(response="Would you like to book flights or accommodation for this trip?"), False
        
        if "flight" in input.lower():
            booking_result = self._travel_planner.book_flight(state["preferences"])
            state["flight_booking"] = booking_result
            return AgentChatResponse(response=f"Great! I've booked your flight. The booking reference is {booking_result['booking_reference']}."), False
        elif "accommodation" in input.lower():
            booking_result = self._travel_planner.book_accommodation(state["preferences"])
            state["accommodation_booking"] = booking_result
            return AgentChatResponse(response=f"Excellent! I've booked your accommodation. The booking reference is {booking_result['booking_reference']}."), False
        else:
            return AgentChatResponse(response="I'm sorry, I didn't understand. Would you like to book flights or accommodation?"), False

    def _finalize_task(self, state: Dict[str, Any], **kwargs: Any) -> None:
        """Finalize task."""
        pass


