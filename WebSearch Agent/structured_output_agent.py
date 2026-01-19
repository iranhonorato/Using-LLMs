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

# O agente (que contém o LLM e a lista de tools, aqui TavilySearch) decide se precisa chamar ferramentas:
# Se necessário, ele chama TavilySearch() para buscar na web (o tool devolve resultados).
agent = create_agent(model=llm, tools=tools_list, response_format=AgentResponse)
# O agente recebe os resultados das ferramentas e constrói um prompt para o LLM que inclui:
# - O que o humano pediu.
# - Os resultados das ferramentas (resumos ou URLs).
# - Instruções de formato: "Retorne um JSON que satisfaça o schema AgentResponse" (isto é automaticamente gerado pelo response_format).

def main():
    print("Inicializando agente")
    
    #  A saída de agent.invoke() geralmente é um dicionário (ou objeto complexo) contendo várias chaves
    result = agent.invoke(
        {
            "messages": HumanMessage(content="Procure pelos melhores cursos de Machine Learning disponíveis na web")
        }
    )
    return result["structured_response"].answer


if __name__ == "__main__":
    output = main()
    print(output)



