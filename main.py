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



def graph_simple():
 # define node functions
    def process(state: SimpleState) -> dict:
        # simple processing logic, for demo purposes
        return {"output": state["input"].upper(), "step": state["step"] + 1}

    # create graph
    graph = StateGraph(SimpleState)

    # add nodes
    graph.add_node("process", process)
    # add edges
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    # execute graph/ compile
    app = graph.compile()





 



















def main():
    print("Hello from learning-langgraph!")


if __name__ == "__main__":
    main()
