from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent

# tool é uma função que um agente pode executar. Pode ser qualquer função que quisermos
# Isso significa que podemos dar ao nosso agente infinitas possibilidades de ações que ele pode executar 
# Por exemplo: Consultar um banco de dados, fazer chamadas via api, fazer consultas na web etc
from langchain.tools import tool 

# Interagir com o agente com mensagens humanas
from langchain_core.messages import HumanMessage 

from langchain_openai import ChatOpenAI


# --------------------------------------------------- Tool-augmented LLM -----------------------------------------------------------
# Biblioteca para buscas na web 
from tavily import TavilyClient
tavily = TavilyClient()

@tool 
def search(query:str) -> str:
    """
    Ferramenta que faz pesquisas na internet 

    Args:
    query: 
        A busca que queremos fazer

    Returns: 
        O reusltado da pesquisa 
    """
    print(f"Buscando por {query}")

    return tavily.search(query=query)

#------------------------------------------------------------------------------------------------------------------




#------------------------------------ ReAct Agent (Reason + Act) --------------------------------------------------
from langchain_tavily import TavilySearch

llm = ChatOpenAI(temperature=0.5, model="gpt-4.1-mini")
tools_list = [TavilySearch()]



# O agente (que contém o LLM e a lista de tools, aqui TavilySearch) decide se precisa chamar ferramentas:
# Se necessário, ele chama TavilySearch() para buscar na web (o tool devolve resultados).

agent = create_agent(model=llm, tools=tools_list) # Agente tem acesso a tools, logo possui capacidade de chamar tools dinamicamente

# O agente recebe os resultados das ferramentas e constrói um prompt para o LLM que inclui:
# - O que o humano pediu.
# - Os resultados das ferramentas (resumos ou URLs).



def main():
    print("Inicializando agente")
    result = agent.invoke({"messages":HumanMessage(content="Procure vagas na área de Quantitativ Finance/Research na região de São Paulo")})
    print(result)
    return result["messages"][-1].content


if __name__ == "__main__":
    output = main()
    print(output)