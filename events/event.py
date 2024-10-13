from enum import Enum, auto

class EventType(Enum):
    GENERATE_ITINERARY_REQUEST = auto()
    ITINERARY_RESPONSE = auto()
    REFINE_ITINERARY_REQUEST = auto()
    FLIGHT_BOOKING_REQUEST = auto()
    ACCOMMODATION_BOOKING_REQUEST = auto()
    BOOKING_CONFIRMATION = auto()