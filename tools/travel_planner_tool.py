from llama_index.core.tools import BaseTool
from llama_index.core.tools.types import ToolMetadata
from duckduckgo_search import DDGS
from llama_index.llms.openai import OpenAI
from typing import Dict, Any
import json
from pydantic import BaseModel, Field

class TravelPlannerSchema(BaseModel):
    query: str = Field(..., description="The travel-related query or request")
    location: str = Field(..., description="The destination or location for the travel query")
    preferences: str = Field(..., description="Any specific preferences or requirements for the travel plan")

class TravelPlannerTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.llm = OpenAI(model="gpt-4o", temperature=0.5)

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="travel_planner",
            description="A tool for planning travel itineraries and providing travel information.",
            fn_schema=TravelPlannerSchema
        )

    def _search_travel_info(self, query: str, max_results: int = 5) -> list:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results

    def _process_search_results(self, results: list) -> str:
        processed_results = []
        for result in results:
            processed_results.append({
                "title": result['title'],
                "snippet": result['body'][:200] + "..."  # Truncate long snippets
            })
        return json.dumps(processed_results, indent=2)

    def __call__(self, query: str, location: str, preferences: str) -> Dict[str, Any]:
        try:
            # Combine query components
            full_query = f"{query} in {location} {preferences}".strip()
            
            # Perform search
            search_results = self._search_travel_info(full_query)
            
            # Process and format results
            processed_results = self._process_search_results(search_results)
            
            # Generate a summary or recommendation based on the results
            summary = f"Based on the search for '{full_query}', here are some travel suggestions:\n\n{processed_results}"
            
            return {
                "status": "success",
                "summary": summary,
                "raw_results": search_results
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"An error occurred while processing the travel query: {str(e)}"
            }