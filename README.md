# Multi-Agent AI Chatbot — Quản lý siêu thị mini

Hệ thống chatbot thông minh, tự động phân loại và trả lời câu hỏi của người dùng thông qua kiến trúc **multi-agent orchestration**: một agent chuyên truy vấn dữ liệu vận hành (SQL) và một agent chuyên tra cứu chính sách (RAG trên PDF), được điều phối bằng **LangGraph**.

## Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Cách chạy](#cách-chạy)
- [Chi tiết các thành phần](#chi-tiết-các-thành-phần)
- [Hạn chế & hướng cải thiện](#hạn-chế--hướng-cải-thiện)

## Tổng quan

Dự án xây dựng một trợ lý ảo cho siêu thị mini, có khả năng:

- Trả lời câu hỏi về **dữ liệu vận hành** (khách hàng, hàng hóa, đơn hàng...) bằng cách tự sinh và thực thi câu truy vấn SQL trên SQL Server.
- Trả lời câu hỏi về **chính sách mua bán sản phẩm** bằng cách tra cứu trong tài liệu PDF thông qua kỹ thuật RAG (Retrieval-Augmented Generation).
- Tự động **định tuyến** (route) câu hỏi đến đúng agent xử lý dựa trên ý định của người dùng.
- **Ghi log** lại mỗi lượt tương tác để phục vụ theo dõi và giám sát.

## Kiến trúc hệ thống

```
START
  │
  ▼
route ──(Command goto)──► "sql" hoặc "pdf"
  │                              │
  ▼                              ▼
sql_agent                    pdf_agent
  │                              │
  └──────────┬───────────────────┘
             ▼
       write_report
             │
             ▼
            END
```

- **route**: dùng LLM với Structured Output (Pydantic) để phân loại câu hỏi thuộc nhóm `sql` hay `pdf`, sau đó điều hướng động bằng `Command(goto=...)` của LangGraph.
- **sql_agent**: agent tự khám phá schema database, tự sinh câu SQL, tự đánh giá tính đúng đắn trước khi thực thi.
- **pdf_agent**: agent tra cứu ngữ cảnh liên quan từ vector store (ChromaDB) rồi trả lời dựa trên nội dung tìm được.
- **write_report**: tóm tắt lại hành động của người dùng và cách hệ thống xử lý, ghi vào file log.

## Công nghệ sử dụng

| Thành phần          | Công nghệ                                                          |
| ------------------- | ------------------------------------------------------------------ |
| Ngôn ngữ            | Python                                                             |
| Orchestration       | LangGraph (`StateGraph`, `Command`)                                |
| Agent framework     | LangChain (`create_agent`)                                         |
| LLM                 | Google Gemini 2.5 Flash (`langchain-google-genai`)                 |
| Database            | SQL Server (SQLAlchemy + pyodbc)                                   |
| Vector store        | ChromaDB                                                           |
| Embedding           | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, chạy local) |
| Đọc PDF             | `PyPDFLoader`                                                      |
| Chunking            | `RecursiveCharacterTextSplitter`                                   |
| Validate dữ liệu    | Pydantic                                                           |
| Cấu hình môi trường | `python-dotenv`                                                    |

## Cài đặt

### 1. Yêu cầu hệ thống

- Python 3.10+
- SQL Server (Express hoặc bản đầy đủ) đã cài đặt và có database `QuanLySieuThiMini`
- ODBC Driver 17 for SQL Server

### 2. Clone dự án và tạo môi trường ảo

```bash
git clone <repo-url>
cd <project-folder>
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux
```

### 3. Cài thư viện

```bash
pip install -r requirements.txt
```

## Cấu hình

Tạo file `.env` ở thư mục gốc với các biến môi trường cần thiết, ví dụ:

```
GOOGLE_API_KEY=your_gemini_api_key
```

Cập nhật thông tin kết nối SQL Server trong code nếu cần (tên server, tên database):

```python
server = r"DESKTOP-S6TLJ08\SQLEXPRESS"
database = "QuanLySieuThiMini"
```

Đặt file tài liệu chính sách PDF (`chinh_sach_mua_ban_san_pham.pdf`) vào thư mục gốc dự án — hệ thống sẽ tự động embed và lưu vào `./chroma_db` ở lần chạy đầu tiên.

## Cách chạy

```bash
python main.py
```

Ví dụ gọi trực tiếp trong code:

```python
result = graph.invoke({
    "user_question": "Có những loại hàng hóa nào?",
    "user_id": 1
})
print(result["messages"][-1])
```

## Chi tiết các thành phần

### SQL Agent — bộ công cụ (tools)

| Tool                            | Chức năng                                                  |
| ------------------------------- | ---------------------------------------------------------- |
| `get_table()`                   | Liệt kê tất cả bảng có trong database                      |
| `get_schema(tables)`            | Lấy tên cột, kiểu dữ liệu và dữ liệu mẫu của bảng chỉ định |
| `get_information(code)`         | Thực thi câu SQL và trả kết quả                            |
| `evaluate(code, user_question)` | Dùng LLM tự đánh giá câu SQL có trả lời đúng câu hỏi không |

Agent được cấu hình (qua system prompt) để **luôn kiểm tra schema trước khi viết SQL**, và **luôn evaluate trước khi thực thi**, nhằm giảm rủi ro sinh câu truy vấn sai.

### RAG Pipeline

1. Đọc PDF chính sách bằng `PyPDFLoader`.
2. Chia nhỏ văn bản thành các đoạn 1000 ký tự, overlap 200 ký tự.
3. Nhúng (embed) từng đoạn bằng model local `all-MiniLM-L6-v2`.
4. Lưu vào ChromaDB tại `./chroma_db` (có cơ chế cache — không embed lại nếu đã tồn tại).
5. Khi truy vấn, tìm 3 đoạn liên quan nhất và lọc bỏ các đoạn có độ liên quan ≤ 0.6.

### Ghi log

Sau mỗi lượt hội thoại, một LLM phụ tóm tắt lại hành động của người dùng (`action`) và cách hệ thống xử lý (`handle`), ghi append vào file `data.txt`.

## Hạn chế & hướng cải thiện

- **Bảo mật SQL**: `get_information` hiện thực thi trực tiếp câu SQL do LLM sinh ra mà chưa có lớp kiểm soát quyền (whitelist câu lệnh, giới hạn chỉ `SELECT`, sandbox). Cần bổ sung trước khi đưa vào môi trường production.
- **Kết nối database**: đang dùng Windows Authentication tới SQL Server cục bộ — cần điều chỉnh để chạy được trên môi trường khác (Linux, cloud).
- **Logging**: đang ghi log dạng text đơn giản, chưa có timestamp hay structured JSON — nên nâng cấp để phục vụ giám sát tốt hơn.

## License

Dự án phục vụ mục đích học tập/demo.
