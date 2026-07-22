import getpass
import os
import re
from typing import TypedDict, List

# This project uses python-dotenv to load API keys from the .env file.
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END


load_dotenv()


def load_env_variable(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Please add {name} to the .env file.")
    return value


def validate_text_field(value: str, field_name: str, min_len: int = 3, max_len: int = 200) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required.")
    text = value.strip()
    if len(text) < min_len or len(text) > max_len:
        raise ValueError(f"{field_name} must be between {min_len} and {max_len} characters.")
    if re.search(r"[<>\"'`;]|\\b(drop|delete|update|insert)\\b", text, re.IGNORECASE):
        raise ValueError(f"{field_name} contains unsafe characters or keywords.")
    return text


def verify_auth_token(expected_token: str) -> None:
    prompt = getpass.getpass("Enter agent auth token: ").strip()
    if prompt != expected_token:
        raise RuntimeError("Invalid agent auth token.")


class ReportState(TypedDict):
    topic: str
    research_notes: List[str]
    summary: str
    draft_report: str
    review_notes: str
    final_report: str


def create_workflow(llm, search_tool):
    def manager_agent_node(state: ReportState):
        state.setdefault("research_notes", [])
        return state

    def research_agent_node(state: ReportState):
        query = state["topic"]
        search_results = search_tool.invoke(query)
        notes = []
        for item in search_results:
            if isinstance(item, dict):
                notes.append(item.get("content") or item.get("title") or str(item))
            else:
                notes.append(str(item))
        state["research_notes"] = notes
        return state

    def summary_agent_node(state: ReportState):
        prompt = f"""
You are a summary agent.
Create a concise, professional summary of the topic below using the research notes.

Topic:
{state['topic']}

Research Notes:
{state['research_notes']}

Return:
1. Short summary
2. 3 key takeaways
"""
        response = llm.invoke(prompt)
        state["summary"] = response.content.strip()
        return state

    def writing_agent_node(state: ReportState):
        prompt = f"""
You are a writing agent.
Write a polished draft report from the summary and research notes.

Topic:
{state['topic']}

Summary:
{state['summary']}

Research Notes:
{state['research_notes']}

Format:
- Introduction
- Main points
- Conclusion
"""
        response = llm.invoke(prompt)
        state["draft_report"] = response.content.strip()
        return state

    def review_agent_node(state: ReportState):
        prompt = f"""
You are a review agent.
Improve the draft report for clarity, structure, and professionalism.

Topic:
{state['topic']}

Draft Report:
{state['draft_report']}

Return a polished review with suggested improvements and a final refined version.
"""
        response = llm.invoke(prompt)
        state["review_notes"] = response.content.strip()
        return state

    def report_manager_agent_node(state: ReportState):
        final_report = f"""# Report

## Topic
{state['topic']}

## Summary
{state['summary']}

## Draft Report
{state['draft_report']}

## Review Notes
{state['review_notes']}
"""
        state["final_report"] = final_report
        return state

    graph = StateGraph(ReportState)
    graph.add_node("manager", manager_agent_node)
    graph.add_node("research", research_agent_node)
    graph.add_node("summary", summary_agent_node)
    graph.add_node("writing", writing_agent_node)
    graph.add_node("review", review_agent_node)
    graph.add_node("finalize", report_manager_agent_node)

    graph.add_edge(START, "manager")
    graph.add_edge("manager", "research")
    graph.add_edge("research", "summary")
    graph.add_edge("summary", "writing")
    graph.add_edge("writing", "review")
    graph.add_edge("review", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def main():
    groq_key = load_env_variable("GROQ_API_KEY")
    tavily_key = load_env_variable("TAVILY_API_KEY")
    auth_token = load_env_variable("AGENT_AUTH_TOKEN")

    verify_auth_token(auth_token)

    llm = ChatGroq(model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"), temperature=0)
    search_tool = TavilySearchResults(max_results=5)
    app = create_workflow(llm, search_tool)

    topic = input("Enter the report topic: ").strip() or "Latest trends in Agentic AI"
    topic = validate_text_field(topic, "Report topic", min_len=5, max_len=150)
    initial_state: ReportState = {
        "topic": topic,
        "research_notes": [],
        "summary": "",
        "draft_report": "",
        "review_notes": "",
        "final_report": "",
    }

    result = app.invoke(initial_state)
    print("\n===== FINAL REPORT =====\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()
