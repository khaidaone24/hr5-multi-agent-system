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
    # --- 1️⃣ Load biến môi trường ---
    load_dotenv()
    DB_LINK = os.getenv("DB_LINK")
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not DB_LINK or not GEMINI_API_KEY:
        raise ValueError("⚠️ Thiếu DB_LINK hoặc GOOGLE_API_KEY trong .env")

    # --- 2️⃣ Cấu hình Postgres MCP Pro (qua uv) ---
    config = {
        "mcpServers": {
            "postgres": {
                "command": "uv",
                "args": [
                    "run",
                    "postgres-mcp",
                    "--access-mode=unrestricted",  # Cho phép đọc/ghi & introspection
                ],
                "env": {
                    "DATABASE_URI": DB_LINK
                },
            }
        }
    }

    print("🚀 Đang khởi tạo MCP Client...")
    client = MCPClient.from_dict(config)
    await client.create_all_sessions()
    print("✅ MCP Client đã kết nối!\n")

    # --- 3️⃣ Đợi MCP Server load schema ---
    print("⏳ Đang đợi MCP server load schema...\n")
    await asyncio.sleep(5)
    session = client.get_session("postgres")

    # --- 4️⃣ Liệt kê tool ---
    tools = await session.list_tools()
    print("🧰 Tool khả dụng:")
    for t in tools:
        print(f" - {t.name}: {t.description}")
    print("\n")

    # --- 5️⃣ Tải danh sách schema & bảng (warm-up introspection) ---
    print("📦 Đang kiểm tra schema và bảng...")
    schema_map = {}

    try:
        schemas = await session.call_tool("list_schemas", {})
        if schemas and schemas.content:
            print("✅ Schema có sẵn:")
            print(schemas.content[0].text)

            # Lấy bảng trong schema 'public'
            tables = await session.call_tool("list_objects", {"schema_name": "public"})
            if tables and tables.content:
                text = tables.content[0].text
                print("\n📊 Bảng có trong schema 'public':")
                print(text)

                # Lưu vào schema_map để Agent hiểu đúng tên bảng
                for line in text.splitlines():
                    if "'name':" in line:
                        name = line.split("'name':")[1].split("'")[1]
                        schema_map[name.lower()] = "public"
    except Exception as e:
        print(f"⚠️ Không thể lấy danh sách schema: {e}")

    # --- 6️⃣ Khởi tạo Gemini LLM ---
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )

    # --- 7️⃣ Tạo MCP Agent ---
    agent = MCPAgent(llm=llm, client=client, max_steps=30)
    print("\n🤖 Agent sẵn sàng nhận truy vấn!")

    # --- 8️⃣ Vòng lặp truy vấn ---
    while True:
        query_text = input("\n💬 Nhập truy vấn (hoặc 'exit' để thoát): ").strip()
        if query_text.lower() in ["exit", "quit"]:
            break

        print("\n⏳ Đang xử lý truy vấn...\n")

        try:
            # Gợi ý LLM: buộc kiểm tra schema trước khi chạy
            hint = (
                "Hãy dùng list_schemas, get_object_details và list_objects để xác định đúng bảng, "
                "ưu tiên dùng các bảng trong schema 'public'. "
                
            )

            result = await agent.run(hint + query_text)
            print(f"\n🎯 Kết quả:\n{result}\n")

        except Exception as e:
            print(f"❌ Lỗi khi xử lý truy vấn: {e}")

    # --- 9️⃣ Đóng kết nối ---
    await client.close_all_sessions()
    print("🔌 Đã đóng kết nối MCP. Hẹn gặp lại!")


if __name__ == "__main__":
    asyncio.run(main())
