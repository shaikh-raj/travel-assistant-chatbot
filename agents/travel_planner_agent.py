
from typing import Dict, Any
from llama_index.core.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
import json
import random
import string
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import uuid
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class TravelPlannerAgent:
    def __init__(self, tools, llm=None):
        self.tools = tools
        self.llm = llm or OpenAI(model="gpt-4o")
        self._memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

    def get_db_connection(self):
        return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)

    def create_session(self, user_preferences):
        session_id = str(uuid.uuid4())
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sessions (session_id, user_preferences) VALUES (%s, %s) RETURNING session_id",
                    (session_id, json.dumps(user_preferences))
                )
                conn.commit()
                return cur.fetchone()['session_id']

    def create_booking(self, session_id, booking_type, booking_details):
        booking_id = self._generate_booking_id()
        status = "confirmed"  # You might want to make this dynamic based on the booking process
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO bookings (booking_id, session_id, booking_type, status, booking_details) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING booking_id",
                    (booking_id, session_id, booking_type, status, json.dumps(booking_details))
                )
                conn.commit()
                return cur.fetchone()['booking_id']

    def generate_itinerary(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Generating itinerary with preferences: {json.dumps(preferences, indent=2)}")
        session_id = self.create_session(preferences)
        self._memory.put(ChatMessage(role=MessageRole.USER, content=f"Generate itinerary request: {json.dumps(preferences)}"))
        itinerary = self._generate_itinerary(preferences)
        print(f"Generated itinerary: {json.dumps(itinerary, indent=2)}")
        return itinerary, session_id

    def book_flight(self, flight_details: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Booking flight with details: {json.dumps(flight_details, indent=2)}")
        self._memory.put(ChatMessage(role=MessageRole.USER, content=f"Flight booking request: {json.dumps(flight_details)}"))
        
        # Extract session_id from flight_details or generate a new one if not present
        session_id = flight_details.get('session_id') or self.create_session(flight_details)
        
        # Create a booking in the database
        booking_id = self.create_booking(
            session_id=session_id,
            booking_type="flight",
            booking_details=flight_details
        )
        
        # Simulate the flight booking process
        booking_result = self._book_flight(flight_details)
        
        print(f"Flight booking result: {json.dumps(booking_result, indent=2)}")
        return {"booking_reference": booking_id, **booking_result}

    def book_accommodation(self, accommodation_details: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Booking accommodation with details: {json.dumps(accommodation_details, indent=2)}")
        self._memory.put(ChatMessage(role=MessageRole.USER, content=f"Accommodation booking request: {json.dumps(accommodation_details)}"))
        
        # Extract session_id from accommodation_details or generate a new one if not present
        session_id = accommodation_details.get('session_id') or self.create_session(accommodation_details)
        
        # Create a booking in the database
        booking_id = self.create_booking(
            session_id=session_id,
            booking_type="accommodation",
            booking_details=accommodation_details
        )
        
        # Simulate the accommodation booking process
        booking_result = self._book_accommodation(accommodation_details)
        
        print(f"Accommodation booking result: {json.dumps(booking_result, indent=2)}")
        return {"booking_reference": booking_id, **booking_result}

    def _generate_itinerary(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        As a travel planning assistant, generate a detailed itinerary based on these preferences:

        {json.dumps(preferences, indent=2)}

        The itinerary should include:
        1. Daily activities and attractions (including any suggested_activities that were accepted)
        2. Suggested restaurants or dining experiences
        3. Transportation recommendations between locations
        4. Any additional tips or suggestions based on the preferences

        Provide a structured itinerary for the entire trip duration.
        """
        response = self.llm.complete(prompt)
        return {"full_itinerary": response.text.strip()}

    def _book_flight(self, flight_details: Dict[str, Any]) -> Dict[str, Any]:
        # Existing implementation
        return {"status": "confirmed"}

    def _book_accommodation(self, accommodation_details: Dict[str, Any]) -> Dict[str, Any]:
        # Existing implementation
        return {"status": "confirmed"}

    def _generate_booking_id(self):
        # Generate a unique 10-character alphanumeric booking ID
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    # Add other database-related methods (get_booking, update_booking, delete_booking) here