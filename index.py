from sqlalchemy import create_engine, text, inspect
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langgraph.types import Command
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from typing import Literal
from operator import add
load_dotenv()
server = r"DESKTOP-S6TLJ08\SQLEXPRESS"
database = "QuanLySieuThiMini"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
uri = (
    f"mssql+pyodbc://{server}/{database}"
    "?driver=ODBC+Driver+17+for+SQL+Server"   # đổi 18 → 17
    "&trusted_connection=yes"
    "&TrustServerCertificate=yes"
)
class State(TypedDict):
  user_question: str
  user_id: str
  messages: Annotated[list[str], add]
  
engine = create_engine(uri)

@tool('get_information')
def get_information(code: str):
  """Retrieve data by code sql after user's question and code is match

  Args:
      code (str): Code sql to retrieve data

  Returns:
      str: relevant data with user's question
  """
  with engine.connect() as conn:
    result = conn.execute(text(code))
    rows = result.fetchall()
    # print(rows)
  return "\n\n".join(str(row) for row in rows)
# print(get_information('select top 3 * from KhachHang'))

@tool('evaluate')
def evaluate(code: str, user_question: str):
    """Use to check if code can handle user's question or not"""
    prompt = f"""You're a master in SQL query evaluation.

Your task is to determine whether the given SQL query correctly and completely answers the user's question.

User's question: {user_question}

SQL query: {code}

Evaluate based on these criteria:
- Does the query select the correct columns needed to answer the question?
- Does the query use the correct tables and joins (if needed)?
- Does the query have correct filter conditions (WHERE clause) matching the question's intent?
- Is the query free of syntax errors?

Respond in this exact format:
Valid: <true or false>
Reason: <brief explanation in 1-2 sentences>
"""
    return llm.invoke(prompt).content

@tool('get_table')
def get_table():
  """ Lấy ra các bảng dữ liệu có trong database """
  inspector = inspect(engine)
  # print(inspector.get_table_names())
  return ", ".join(inspector.get_table_names())
# print(get_table())


@tool('get_schema')
def get_schema(tables: list[str]):
  """Lấy schema (tên cột, kiểu dữ liệu) và vài dòng mẫu của các bảng được chỉ định.

    Args:
        tables (list[str]): Danh sách tên các bảng cần lấy thông tin schema.

    Returns:
        str: Thông tin schema và dữ liệu mẫu của các bảng, dạng text.
    """
  result = []
  inspector = inspect(engine)
  with engine.connect() as conn:
    for table in tables:
      cols = inspector.get_columns(table)
      col_info = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
      example = conn.execute(text(f'select top 3 * from {table}')).fetchall()
      result.append(f"Table {table}:\nColumns: {col_info}\nSample rows: {example}")
  return "\n\n".join(result)
# print(get_schema(get_table()))

sql_agent = create_agent(
  model=llm,   # hoặc "anthropic:claude-sonnet-4-6", tùy provider bạn dùng
  tools=[get_information, get_table, get_schema, evaluate],
  system_prompt="Bạn là trợ lý truy vấn dữ liệu SQL Server. Luôn kiểm tra schema trước khi viết code, và dùng evaluate trước khi execute."
)
class Log(BaseModel):
  action: str = Field(description = "Action's User")
  handle: str = Field(description = "How the agent hadle this user's request")
llm_with_structure = llm.with_structured_output(Log)
def create_store():
  embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2"
  )
  collection_name = "my_pdf_collection"
  persist_directory="./chroma_db"
  if os.path.exists("./chroma_db"):
    # Đã có DB rồi → load lại, không embed lại từ đầu
    return Chroma(
      collection_name=collection_name,
      embedding_function=embeddings,
      persist_directory=persist_directory
    )
  
  loader = PyPDFLoader("chinh_sach_mua_ban_san_pham (1).pdf")
  pages = loader.load()  # trả về list các Document, mỗi phần tử = 1 trang



  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  chunks = splitter.split_documents(pages)
  
  vector_store = Chroma.from_documents(
      documents=chunks,
      embedding=embeddings,
      collection_name=collection_name,
      persist_directory=persist_directory 
  )
  return vector_store
vector_store = create_store()
@tool
def rag_tool(question: str) -> str:
  """Tìm kiếm thông tin liên quan trong tài liệu sản phẩm dựa trên câu hỏi của khách hàng."""
  relevant_information = vector_store.similarity_search_with_relevance_scores(question, k = 3)
  relevant_information = "\n\n".join([ri.page_content for ri, score in relevant_information if score > 0.6])
  return relevant_information



def write_report(state):
  user_question = state['user_question']
  user_id = state['user_id']
  ai_response = state['messages'][-1]
  context = f"User's question: {user_question}, AI response: {ai_response}"
  logs = llm_with_structure.invoke(context)
  with open("data.txt", "a", encoding="utf-8") as f:
    f.write(f'{user_id} {logs.action} {logs.handle}\n')
pdf_agent = create_agent(
  model=llm,   # hoặc "anthropic:claude-sonnet-4-6", tùy provider bạn dùng
  tools=[rag_tool],
  system_prompt="Bạn là trợ lí AI hỗ trợ truy xuất thông tin có trong file pdf bằng rag_tool"
)

def sql(state):
    user_question = state['user_question']
    response = sql_agent.invoke({"messages": [{"role": "user", "content": user_question}]})
    return {'messages': [response["messages"][-1].content]}   # sửa cả cách lấy kết quả
def pdf(state):
    user_question = state['user_question']
    response = pdf_agent.invoke({"messages": [{"role": "user", "content": user_question}]})
    return {'messages': [response["messages"][-1].content]}
class Route(BaseModel):
  pdf_or_sql: Literal["sql", "pdf"] = Field(
      description="Trả về 'sql' nếu câu hỏi cần truy vấn database, 'pdf' nếu cần tra cứu tài liệu chính sách"
  )
def route(state):
  user_question = state['user_question']
  llm_with_structure = llm.with_structured_output(Route)
  pdf_or_sql = llm_with_structure.invoke(user_question)
  return Command(goto = pdf_or_sql.pdf_or_sql)



builder = StateGraph(State)

builder.add_node("route", route)
builder.add_node("sql", sql)
builder.add_node("pdf", pdf)
builder.add_node("write_report", write_report)

builder.add_edge(START, "route")
# route dùng Command(goto=...) nên không cần add_conditional_edges,
# nhưng sql/pdf → write_report vẫn là static edge bình thường
builder.add_edge("sql", "write_report")
builder.add_edge("pdf", "write_report")
builder.add_edge("write_report", END)

graph = builder.compile()

result = graph.invoke({"user_question": "Có những loại hàng hóa nào?", 'user_id': 1})
print(result['messages'][-1])