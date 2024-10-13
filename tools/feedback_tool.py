from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from typing import Dict, Any, Optional, List
import json
from pydantic import BaseModel, Field
import logging
from datetime import datetime

class TravelPreferences(BaseModel):
    destination: Optional[str] = Field(None, description="The desired travel destination")
    start_date: Optional[str] = Field(None, description="The start date of the trip (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="The end date of the trip (YYYY-MM-DD)")
    budget: Optional[str] = Field(None, description="The budget for the trip")
    accommodation_type: Optional[str] = Field(None, description="The preferred type of accommodation")
    activities: Optional[List[str]] = Field(None, description="List of preferred activities")
    transportation: Optional[str] = Field(None, description="Preferred mode of transportation")
    suggested_activities: Optional[List[str]] = Field(None, description="List of suggested activities")
    need_extra_suggestions: bool = Field(True, description="Whether the user wants additional suggestions")

class FeedbackTool:
    def __init__(self):
        self.llm = OpenAI(model="gpt-4o", temperature=0.5)
        logging.debug("FeedbackTool initialized")

    def process_feedback(self, feedback: str, current_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback and update travel preferences."""
        schema_fields = list(TravelPreferences.__fields__.keys())
        
        prompt = f"""
        Analyze the user feedback and update the travel preferences:
        Current preferences: {json.dumps(current_preferences, indent=2)}
        User feedback: '{feedback}'

        Instructions:
        1. Update existing preferences or add new ones based on the feedback.
        2. Remove irrelevant preferences by setting them to null.
        3. For suggested activities, update the 'suggested_activities' list.
        4. Convert relative dates to actual dates (YYYY-MM-DD). Today is {datetime.now().strftime('%Y-%m-%d')}.
        5. If user indicates no more suggestions needed (e.g., "no", "that's enough"), set 'need_extra_suggestions' to false.
        6. Always include 'need_extra_suggestions' in your response.

        Respond with a JSON object containing only the following fields from the TravelPreferences schema:
        {', '.join(schema_fields)}

        Ensure all dates are in YYYY-MM-DD format and the JSON is properly formatted with no ``` json or anything in the beginning.
        """

        response = self.llm.complete(prompt)
        print(f"LLM feedback response: {response.text}")
        try:
            updated_prefs = json.loads(response.text)
            # Validate that only schema fields are present
            for key in list(updated_prefs.keys()):
                if key not in schema_fields:
                    logging.warning(f"Unexpected field '{key}' in LLM response. Removing.")
                    del updated_prefs[key]
            return updated_prefs
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response: {response.text}")
            raise ValueError("Failed to parse updated preferences from LLM response")

    def update_preferences(self, feedback: str, current_preferences: str) -> Dict[str, Any]:
        """Update travel preferences based on user feedback."""
        logging.debug(f"FeedbackTool called with feedback: {feedback}, current_preferences: {current_preferences}")
        try:
            current_prefs = json.loads(current_preferences) if current_preferences else {}
            
            # Process feedback and update preferences
            updated_prefs_dict = self.process_feedback(feedback, current_prefs)
            updated_prefs = TravelPreferences.parse_obj(updated_prefs_dict)
            
            # Compare and summarize changes
            changes = []
            for field in TravelPreferences.__fields__:
                old_value = current_prefs.get(field)
                new_value = getattr(updated_prefs, field)
                if old_value != new_value:
                    if new_value is None:
                        changes.append(f"Removed '{field}'")
                    elif old_value is None:
                        changes.append(f"Added '{field}': {new_value}")
                    else:
                        changes.append(f"Updated '{field}': {old_value} -> {new_value}")
            
            change_summary = "\n".join(changes)
            
            return {
                "status": "success",
                "updated_preferences": updated_prefs.dict(exclude_none=True),
                "change_summary": change_summary
            }
        except ValueError as e:
            logging.error(f"Invalid input: {str(e)}")
            return {
                "status": "error",
                "message": f"Invalid input: {str(e)}"
            }
        except Exception as e:
            logging.error(f"Error in FeedbackTool: {str(e)}")
            return {
                "status": "error",
                "message": f"An error occurred while processing the feedback: {str(e)}"
            }

# Create the FunctionTool
feedback_tool = FunctionTool.from_defaults(
    fn=FeedbackTool().update_preferences,
    name="feedback_processor",
    description="A tool for processing user feedback and updating travel preferences.",
)