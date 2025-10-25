import asyncio
import os
try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not installed")
    def load_dotenv(): pass

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    print("Warning: langchain-google-genai not installed")
    ChatGoogleGenerativeAI = None

try:
    from mcp_use import MCPAgent, MCPClient
except ImportError:
    print("Warning: mcp_use not installed")
    MCPAgent = None
    MCPClient = None


async def main():
    # --- 1. Load biến môi trường ---
    load_dotenv()
    DB_LINK = os.getenv("DB_LINK")
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not DB_LINK or not GEMINI_API_KEY:
        raise ValueError("❌ Thiếu DB_LINK hoặc GOOGLE_API_KEY trong file .env")

    # --- 2. Cấu hình MCP Server cho Postgres ---
    config = {
        "mcpServers": {
            "postgres": {
                "command": "uv",
                "args": [
                    "run",
                    "postgres-mcp",
                    "--access-mode=unrestricted",
                ],
                "env": {"DATABASE_URI": DB_LINK},
            }
        }
    }

    print("Dang khoi tao MCP Client...")
    client = MCPClient.from_dict(config)
    await client.create_all_sessions()
    print("MCP Client da ket noi!\n")

    print("Dang doi MCP server load schema...\n")
    await asyncio.sleep(5)
    session = client.get_session("postgres")

    # --- 3. Liệt kê tool ---
    tools = await session.list_tools()
    print("Tool kha dung:")
    for t in tools:
        print(f" - {t.name}: {t.description}")
    print("\n")

    # --- 4. Kiểm tra schema & bảng ---
    print("Dang kiem tra schema va bang...")
    try:
        schemas = await session.call_tool("list_schemas", {})
        print(" Schema co san:")
        print(schemas.content[0].text)

        tables = await session.call_tool("list_objects", {"schema_name": "public"})
        print("\n Bang co trong schema 'public':")
        print(tables.content[0].text)
    except Exception as e:
        print(f"Khong the lay danh sach schema: {e}")

    # --- 5. Tạo Gemini LLM ---
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )

    # --- 6. Tạo MCP Agent ---
    agent = MCPAgent(llm=llm, client=client, max_steps=30)
    print("\nAgent san sang nhan truy van!")

    # --- 7. Lấy thông tin chi tiết các bảng quan trọng ---
    print("Dang lay thong tin chi tiet cac bang...")
    table_details = {}
    important_tables = ['nhan_vien', 'phong_ban', 'chuc_vu', 'cong_viec', 'du_an']
    
    for table_name in important_tables:
        try:
            details = await session.call_tool("get_object_details", {
                "schema_name": "public",
                "object_name": table_name
            })
            if details and details.content:
                table_details[table_name] = details.content[0].text
                print(f"Da lay thong tin bang {table_name}")
        except Exception as e:
            print(f"Khong the lay thong tin bang {table_name}: {e}")
    
    # --- 8. Gợi ý ban đầu cho AI với thông tin chi tiết ---
    schema_info = ""
    for table_name, details in table_details.items():
        try:
            import json
            detail_obj = json.loads(details)
            if 'columns' in detail_obj:
                columns = [col['column'] for col in detail_obj['columns']]
                schema_info += f"`{table_name}` có các cột: {', '.join(columns)}. "
        except:
            schema_info += f"`{table_name}` (thông tin chi tiết có sẵn). "
    
    hint = (
        "Ban dang lam viec voi co so du lieu quan ly nhan su. "
        f"Cau truc bang: {schema_info}"
        "QUAN TRONG: Luon su dung dung ten cot nhu da liet ke o tren. "
        "Vi du: nhan_vien co cot 'luong_co_ban' (KHONG phai 'luong'), phong_ban co cot 'id' (KHONG phai 'id_phong_ban'). "
        "Ban co the dung get_object_details de xem chi tiet cot neu can. "
        "Hay tra loi bang tieng Viet ngan gon, kem ket qua truy van neu co. "
    )

    # --- 9. Vòng lặp truy vấn ---
    while True:
        query_text = input("\nNhap truy van (hoac 'exit' de thoat): ").strip()
        if query_text.lower() in ["exit", "quit"]:
            break

        print("\nDang xu ly truy van...\n")
        try:
            result = await agent.run(hint + query_text)
            print(f"\nKet qua:\n{result}\n")
        except Exception as e:
            print(f"Loi khi xu ly truy van: {e}")

    # --- 10. Đóng kết nối ---
    await client.close_all_sessions()
    print("Da dong ket noi MCP. Hen gap lai!")


if __name__ == "__main__":
    asyncio.run(main())
