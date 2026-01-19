from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool 
from langchain_core.messages import HumanMessage 
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema para registrar a fonte de consulta usada pelo agente"""
    url:str = Field(description="A URL da fonte")


class AgentResponse(BaseModel):
    """Schema para o return do agente com respostas e fontes de consulta"""
    answer:str = Field(description="A resposta do agente após as querys")
    sources: List[Source] = Field(default_factory=list, description="Lista das fontes utilizadas para gerar a resposta")


llm = ChatOpenAI(temperature=0.5, model="gpt-4.1-mini")
tools_list = [TavilySearch()]
agent = create_agent(model=llm, tools=tools_list, response_format=AgentResponse)


def main():
    print("Inicializando agente")
    result = agent.invoke(
        {
            "messages": HumanMessage(content="Procure pelos melhores cursos de Machine Learning disponíveis na web")
        }
    )
    return result["structured_response"].answer


if __name__ == "__main__":
    output = main()
    print(output)
