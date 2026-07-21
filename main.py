from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from typing import TypedDict
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)




# define simple state
class SimpleState(TypedDict):
    input: str
    output: str
    step: int







 



















def main():
    print("Hello from learning-langgraph!")


if __name__ == "__main__":
    main()
