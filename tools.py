from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from retriever import extract_text

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)

search_tool = DuckDuckGoSearchRun()