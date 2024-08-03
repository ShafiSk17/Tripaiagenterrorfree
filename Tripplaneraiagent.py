# main.py
import streamlit as st
import os
from crewai import Crew, Process, Agent, Task
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# Streamlit UI components
st.title("Trip Planner App 💬")

api_key = st.text_input("Enter Your API key")
from_place = st.text_input("Enter the starting location (From):")
to_place = st.text_input("Enter the destination location (To):")
travel_date = st.date_input("Enter the date of journey:")

# Initialize session state messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Submit button
if st.button("Plan Trip"):
    if api_key and from_place and to_place and travel_date:
        # Setup LLM
        openai_llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=1500,
        )

        # Avatar images
        avators = {
            "Travel Agency Manager": "https://cdn-icons-png.flaticon.com/512/320/320336.png",
            "Local Tourist Guide": "https://cdn-icons-png.flaticon.com/512/2203/2203675.png",
            "Transport & Accommodation Travel Agent": "https://cdn-icons-png.flaticon.com/512/9408/9408201.png",
            "Manager": "https://cdn-icons-png.flaticon.com/512/305/305694.png"
        }

        # Custom callback handler
        class MyCustomHandler(BaseCallbackHandler):
            def __init__(self, agent_name: str) -> None:
                self.agent_name = agent_name

            def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
                st.session_state.messages.append({"role": "assistant", "content": inputs['input']})
                st.write(inputs['input'])

            def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
                st.session_state.messages.append({"role": self.agent_name, "content": outputs['output']})
                st.write(outputs['output'])

        # Agent setup
        agency_manager = Agent(
            role='Travel Agency Manager',
            goal='To greet customers, and manage the customers travel requirements',
            verbose=True,
            backstory="A manager in a reputed travel agency, which helps users to recommend and resolve all the requirements of users",
            llm=openai_llm,
            callbacks=[MyCustomHandler("Travel Agency Manager")],
            tools=[DuckDuckGoSearchRun()]
        )

        local_travel_agent = Agent(
            role='Local Tourist Guide',
            goal='To make travellers visit all the local tourist spots based on day itinerary',
            verbose=True,
            backstory="A famous local tourist guide who can speak multiple languages and has a very good idea about all the tourist spots and the history about the spots",
            llm=openai_llm,
            callbacks=[MyCustomHandler("Local Tourist Guide")],
            tools=[DuckDuckGoSearchRun()]
        )

        transport_accomodation_agent = Agent(
            role='Transport & Accommodation Travel Agent',
            goal='To provide travel recommendations based on weather, and accommodation details',
            verbose=True,
            backstory="You have all details about travel options, accommodation options for user requirements which are not too expensive",
            llm=openai_llm,
            callbacks=[MyCustomHandler("Transport & Accommodation Travel Agent")],
            tools=[DuckDuckGoSearchRun()]
        )

        manager_agent = Agent(
            role="Manager",
            goal="To oversee and coordinate all the travel planning activities, ensuring smooth execution and optimal results.",
            verbose=True,
            backstory="A highly experienced travel manager responsible for coordinating between various agents to deliver the best travel plan.",
            llm=openai_llm,
            callbacks=[MyCustomHandler("Manager")]
        )

        # Task definitions
        def travels_recommendations(agent, from_place, to_place, travel_date):
            return Task(
                description=f"""
                    Collect and summarize all details about travel for user from {from_place} to {to_place}
                    Pay special attention to any significant events happening in {to_place} around this date {travel_date}
                    Your final answer MUST be a report that includes a comprehensive summary of the latest best deals.
                    Make sure to use the most recent data as possible.
                """,
                agent=agent,
                llm=openai_llm,
                expected_output='report of summary of tourist spots based on user requirements'
            )

        def local_places(agent, to_place, travel_date):
            return Task(
                description=f"""
                    Extract all the best tourist spots in {to_place}, based on the weather or season.
                    Try to include a few vendor details like car rentals, bike rentals in the local area.
                    Pay special attention to any significant events happening in {to_place} around this date {travel_date}
                    Your final answer MUST be a report that includes a comprehensive summary of the latest events or places a user can visit without any problems.
                    Make sure to use the most recent data as possible.
                """,
                agent=agent,
                llm=openai_llm,
                expected_output='report of all required details for a user while travelling to new places for vacation, make sure the cost you provide is in Rupees'
            )

        def transport_accomodation(agent, from_place, to_place, travel_date):
            return Task(
                description=f"""
                    Extract all the best deals for transport in all modes like flights or trains from {from_place} to {to_place}.
                    Also, get the best accommodation details in {to_place} with ratings above 4 stars around this date {travel_date}
                    Your final answer MUST be a report that includes a comprehensive summary of the transport details and accommodation options.
                    Make sure to use the most recent data as possible.
                """,
                agent=agent,
                llm=openai_llm,
                expected_output='report of best deals for travel and accommodation as per user requirements, make sure the cost you provide is in Rupees'
            )

        # Initialize Crew
        def init_crew(from_place, to_place, travel_date):
            crew = Crew(
                agents=[agency_manager, local_travel_agent, transport_accomodation_agent],
                tasks=[
                    travels_recommendations(agency_manager, from_place, to_place, travel_date),
                    local_places(local_travel_agent, to_place, travel_date),
                    transport_accomodation(transport_accomodation_agent, from_place, to_place, travel_date)
                ],
                llm=openai_llm,
                process=Process.hierarchical,
                manager_agent=manager_agent
            )
            return crew.kickoff()

        # Run the process and display the result
        try:
            result = init_crew(from_place, to_place, travel_date)
            st.session_state.messages.append({"role": "assistant", "content": str(result)})
            st.write(str(result))
        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.error("Please fill in all the fields.")
