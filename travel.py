import os
import datetime
import operator
import os
import uuid

from typing import Annotated, TypedDict, Optional
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
import serpapi
# Tool definitions
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import gradio as gr
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os


# Load environment variables from a .env file

from dotenv import load_dotenv
load_dotenv()

CURRENT_YEAR = datetime.datetime.now().year

TOOLS_SYSTEM_PROMPT = f"""You are a smart travel agency. Use the tools to look up information.
You are allowed to make multiple calls (either together or in sequence).
Only look up information when you are sure of what you want.
The current year is {CURRENT_YEAR}. If you don't get flights, try searching till you find them!
If you need to look up some information before asking a follow up question, you are allowed to do that!
In your output, include links to hotel websites and flight websites (if possible),
the logo of the hotel and the logo of the airline company (if possible),
and always include both the price of the flight and the price of the hotel (with currency).
For example, for hotels:
    Rate: $581 per night
    Total: $3,488
"""

# define flight tool
class FlightsInput(BaseModel):
    departure_airport:Optional[str] = Field(description="Departure airport code (IATA)")
    arrival_airport:Optional[str] = Field(description="Arrival airport code (IATA)")
    outbound_date:Optional[str] = Field(description="Outbound date (YYYY-MM-DD)")
    return_date:Optional[str] = Field(description="Return date (YYYY-MM-DD)")
    adults:Optional[int] = Field(1,description="Number of adults (default 1)")
    children:Optional[int] = Field(0,description="Number of children (default 0)")
    infants_in_seat: Optional[int] = Field(0, description='Number of infants in seat (default 0)')
    infants_on_lap: Optional[int]   = Field(0, description='Number of infants on lap (default 0)')

class FlightsInputSchema(BaseModel):
    params:FlightsInput

@tool(args_schema=FlightsInputSchema)
def flights_finder(params:FlightsInput):
    """
    Find flights using the Google Flights engine (via SerpAPI).
    Returns:
        dict or str: Flight search results or error message.
    """
    query = {
        'api_key': os.environ.get('SERPAPI_API_KEY'),
        'engine': 'google_flights',
        'hl': 'en',
        'gl': 'us',
        'departure_id': params.departure_airport,
        'arrival_id': params.arrival_airport,
        'outbound_date': params.outbound_date,
        'return_date': params.return_date,
        'currency': 'USD',
        'adults': params.adults,
        'infants_in_seat': params.infants_in_seat,
        'stops': '1',
        'infants_on_lap': params.infants_on_lap,
        'children': params.children
    }
    try:
        search = serpapi.search(query)
        return search.data['best_flights']
    except Exception as e:
        return str(e)


# define hotel tool
class HotelsInput(BaseModel):
    q: str = Field(description='Location for hotels (e.g., "New York")')
    check_in_date: str  = Field(description='Check-in date (YYYY-MM-DD)')
    check_out_date: str = Field(description='Check-out date (YYYY-MM-DD)')
    sort_by: Optional[str] = Field(8, description='Sorting parameter (default=8 for rating)')
    adults: Optional[int]   = Field(1, description='Number of adults (default 1)')
    children: Optional[int] = Field(0, description='Number of children (default 0)')
    rooms: Optional[int]    = Field(1, description='Number of rooms (default 1)')
    hotel_class: Optional[str] = Field(None, description='Filter by hotel class (e.g., "3" or "4")')

class HotelsInputSchema(BaseModel):
    params:HotelsInput

@tool(args_schema=HotelsInputSchema)
def hotels_finder(params:HotelsInput):
    """
    Find hotels using the Google Hotels engine (via SerpAPI).
    Returns:
        list or str: Up to 5 hotel property dicts or error message.
    """
    query = {
        'api_key': os.environ.get('SERPAPI_API_KEY'),
        'engine': 'google_hotels',
        'hl': 'en',
        'gl': 'us',
        'q': params.q,
        'check_in_date': params.check_in_date,
        'check_out_date': params.check_out_date,
        'currency': 'USD',
        'adults': params.adults,
        'children': params.children,
        'rooms': params.rooms,
        'sort_by': params.sort_by,
        'hotel_class': params.hotel_class
    }
    try:
        search = serpapi.search(query)
        data = search.data
        return data['properties'][:5]
    except Exception as e:
        return str(e)

# define tool
TOOLS = [flights_finder,hotels_finder]

# define Agent State
class AgentState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]

# build agent
class Agent():
    def __init__(self):
        self.tools = {t.name: t for t in TOOLS}
        self.tools_llm = ChatOpenAI(model="gpt-4o").bind_tools(TOOLS)
        builder = StateGraph(AgentState)
        builder.add_node("call_tools_llm",self.call_tools_llm)
        builder.add_node("invoke_tools",self.invoke_tools)
        builder.add_node("email_sender",self.email_sender)
        builder.set_entry_point("call_tools_llm")
        builder.add_conditional_edges(
            "call_tools_llm",
            Agent.conditions,
            {"email_sender":"email_sender","more_tools":"invoke_tools"}
        )
        builder.add_edge("invoke_tools","call_tools_llm")
        builder.add_edge("email_sender",END)
        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory,interrupt_before=['email_sender'])

    @staticmethod
    def conditions(state: AgentState):
        result = state["messages"][-1]
        if len(result.tool_calls) == 0:
            return "email_sender"
        else:
            return "more_tools"
        
    def call_tools_llm(self,state: AgentState):
        messages = [SystemMessage(content=TOOLS_SYSTEM_PROMPT)] + state["messages"]
        messages = self.tools_llm.invoke(messages)
        return {"messages":[messages]}
    
    def invoke_tools(self, state: AgentState):
        message = state["messages"][-1]
        results = []
        for t in message.tool_calls:
            if t["name"] not in self.tools:
                results.append(
                    ToolMessage(tool_call_id=t['id'], name=t['name'], content="Bad tool name")
                )
            else:
                messages = self.tools[t["name"]].invoke(t["args"])
                results.append(
                    ToolMessage(tool_call_id = t["id"],name=t["name"],content=str(messages))
                )
        return {"messages":results}
    def email_sender(self, state: AgentState):
        return {'messages': []}



# via Gmail to send

def send_html_email(travel_html: str, sender: str, receiver: str, subject: str) -> str:
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = sender
        msg["To"] = receiver
        msg["Subject"] = subject

        html_part = MIMEText(travel_html, "html")
        msg.attach(html_part)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, os.environ["GMAIL_APP_PASSWORD"])
            server.sendmail(sender, receiver, msg.as_string())

        return "Email sent successfully via Gmail SMTP!"

    except Exception as e:
        return f"Error sending email: {e}"

# create agent
agent = Agent()

# Gradio callables 
def process_query_gradio(user_query: str) -> str:
    """
    Run the agent on the given travel query string, return the final text output.
    """
    thread_id = str(uuid.uuid4())
    # Create a single HumanMessage containing the entire query
    message = [HumanMessage(content=user_query)]
    config = {'configurable': {'thread_id': thread_id}}

    state = agent.graph.invoke({"messages":message}, config=config)
    return state["messages"][-1].content


def process_email_gradio(travel_info: str, sender: str, receiver: str, subject: str) -> str:
    """
    Take the travel_info (HTML or plain text from the agent),
    plus sender/receiver/subject, and send via SendGrid.
    """
    if not sender or not receiver or not subject or not travel_info:
        return "Error: All fields are required."
    return send_html_email(travel_info, sender, receiver, subject)


# ===== 6) Build the Gradio Interface =====
with gr.Blocks() as demo:
    gr.Markdown("# AI Travel Agent")
    gr.Markdown("Enter a travel query below (e.g. “Flights from New York to London June 10–15, and 4-star hotels”).")

    # Textbox to accept the user’s travel query
    query_input = gr.Textbox(lines=3, placeholder="Type your travel query here…", label="Travel Query")
    query_button = gr.Button("Get Travel Information")
    travel_output = gr.Markdown("", label="Travel Info (Agent’s Response)")

    # When clicked, run process_query_gradio and show result in travel_output
    query_button.click(fn=process_query_gradio, inputs=query_input, outputs=travel_output)

    gr.Markdown("---\n## Send the Above Info via Email")
    gr.Markdown("Enter sender, receiver, and subject. The email body will be exactly what the agent printed above.")
    sender_input   = gr.Textbox(label="Sender Email")
    receiver_input = gr.Textbox(label="Receiver Email")
    subject_input  = gr.Textbox(label="Subject", value="Travel Information")
    email_button   = gr.Button("Send Email")
    email_status   = gr.Textbox(label="Email Status / Error")

    # When clicked, run process_email_gradio using travel_output plus the three email fields
    email_button.click(
        fn=process_email_gradio,
        inputs=[travel_output, sender_input, receiver_input, subject_input],
        outputs=email_status
    )
# Launch Gradio app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)

