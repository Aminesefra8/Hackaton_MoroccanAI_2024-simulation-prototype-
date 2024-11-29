

import operator
from typing import List, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, START, StateGraph

from typing import Optional
from pydantic import BaseModel, Field


class Info_Type(BaseModel):
    # Description of bad mental health with details
    mental_health_description: str = Field(
        description=(
            "A detailed description of the user's bad mental health states, "
            "including symptoms, causes, behaviors, and any other information related to periods "
            "where the user experienced poor mental health. If no bad mental health states are detected, leave empty."
        )
    )
    
    # Boolean indicating if bad mental health is detected (past or present)
    is_badmental_health: str = Field(
        description=(
            "A boolean value indicating whether the user has ever experienced bad mental health, "
            "whether in the past or present. Set to 'True' if the description includes any mention of "
            "issues or symptoms of poor mental health; otherwise, 'False'."
        ),
        enum=["True", "False"]
    )
    
    # Boolean to indicate whether the user is currently cured of mental health issues
    Is_mentally_cured: str = Field(
        description=(
            "A boolean value indicating whether the user is currently cured of their mental health issues. "
            "Set to 'True' if the user has completely recovered and is mentally healthy; otherwise, set to 'False'."
        ),
        enum=["True", "False"]
    )
    

    

#Chains 

def get_initial_summary_chain(llm):
    summarize_prompt = ChatPromptTemplate(
        [
            ("human", "Write a concise summary of the following: {context}"),
        ]
    )
    return summarize_prompt | llm | StrOutputParser()


# Function to create the refine summary chain
def get_refine_summary_chain(llm):
    refine_template = """
    Produce a final summary.

    Existing summary up to this point:
    {existing_answer}

    New context:
    ------------
    {context}
    ------------

    Given the new context, refine the original summary.
    """
    refine_prompt = ChatPromptTemplate([("human", refine_template)])
    return refine_prompt | llm | StrOutputParser()


def extraction_chain(llm):

    extract_prompt = ChatPromptTemplate.from_template(
    """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Info_Type' function.

    Passage:
    {input}
    """
    )

    return extract_prompt | llm 



class State(TypedDict):
    contents: List[str]
    index: int
    summary: str
    extract_info_output : dict
    output_dict : dict
    extra_args: dict 
    

# Define the initial summary generation node
async def generate_initial_summary(state: State, config: RunnableConfig):
    llm = state["extra_args"]["llm"]
    initial_summary_chain = get_initial_summary_chain(llm)
    summary = await initial_summary_chain.ainvoke(
        {"context": state["contents"][0]},
        config,
    )
    return {"summary": summary, "index": 1}


# Define the refinement node
async def refine_summary(state: State, config: RunnableConfig):
    llm = state["extra_args"]["llm"] 
    refine_summary_chain = get_refine_summary_chain(llm)
    content = state["contents"][state["index"]]
    summary = await refine_summary_chain.ainvoke(
        {"existing_answer": state["summary"], "context": content},
        config,
    )
    return {"summary": summary, "index": state["index"] + 1}

# Conditional logic to continue refinement or exit
def should_refine(state: State) -> Literal["refine_summary", "extract_info_node"]:
    if state["index"] >= len(state["contents"]):
        return "extract_info_node"
    else:
        return "refine_summary"


async def extract_info_summary(state: State, config: RunnableConfig):
    llm = state["extra_args"]["llm"]
    llm_with_structured_output = llm.with_structured_output(Info_Type)
    extract_info_chain = extraction_chain(llm_with_structured_output)
    summary = state["summary"]
    extraction = await extract_info_chain.ainvoke(
        {"input": summary},
        config,
    )
    return {"extract_info_output": extraction}



def output_final(state: State):

    output_dict = {
        "bad_mental_health_profile": state["extract_info_output"].mental_health_description,
        "is_badmental_health": state["extract_info_output"].is_badmental_health,
        "Is_mentally_cured": state["extract_info_output"].Is_mentally_cured,
        "motivation": state["extract_info_output"].motivation
    }

    return {"output_dict": output_dict}


# Function to create and return the graph
def create_app():

    graph = StateGraph(State)
    
    # Add nodes with the appropriate dependencies
    graph.add_node("generate_initial_summary", generate_initial_summary)
    graph.add_node("refine_summary", refine_summary)
    graph.add_node("extract_info_node", extract_info_summary)
    graph.add_node("output_final", output_final)

    # Add edges
    graph.add_edge(START, "generate_initial_summary")
    graph.add_conditional_edges("generate_initial_summary", should_refine)
    graph.add_conditional_edges("refine_summary", should_refine)
    graph.add_edge("extract_info_node", "output_final")
    graph.add_edge("output_final", END)

    # Compile the graph
    return graph.compile()
