
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import START , END , StateGraph, MessagesState
from pydantic import BaseModel, Field
from typing import Annotated, TypedDict
import os

load_dotenv()
class ChatSchema(TypedDict):
	message : Annotated[str , Field(description="enter message" )  ]
	res : Annotated[str, Field(description="enter res ")]

temp = {
	"message" : "kirat@gmail.com",
	"res" : "23"
}
final_person = ChatSchema(**temp)
print(final_person)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)
state = {}
def initiate_chat(state: ChatSchema):
    mes = state["message"]
    res = llm.invoke(mes)
    return {"res": res.content}


graph = StateGraph(ChatSchema)

graph.add_node("chat_response" , initiate_chat)
graph.add_edge(START , "chat_response")
graph.add_edge("chat_response" , END)

model = graph.compile()

state = {
    "message": "What is the capital of India?"
}
config = {"configurable" : {"thread_id" : 1}}
for chunk , metadata in model.stream(
      {"message" : "what is the code of two sum ? "},
		config=config,
        stream_mode= "messages"
):
      if chunk.content:
            print(chunk.content[-1]['text'])
# response = model.invoke(state)/

# print(response["res"][-1]["text"])