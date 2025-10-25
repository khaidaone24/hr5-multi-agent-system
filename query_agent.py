import asyncio
import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from mcp_use import MCPAgent, MCPClient
except ImportError:
    print("Warning: mcp_use not installed")
    # Create dummy classes
    class MCPAgent:
        def __init__(self, *args, **kwargs):
            pass
        async def run(self, *args, **kwargs):
            return "Mock query result"
    class MCPClient:
        def __init__(self, *args, **kwargs):
            pass

class QueryAgent:
    """
    Query Agent - Xử lý truy vấn cơ sở dữ liệu thông qua MCP
    """
    
    def __init__(self):
        load_dotenv()
        self.DB_LINK = os.getenv("DB_LINK")
        self.GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        if not self.GEMINI_API_KEY:
            print(" Warning: GOOGLE_API_KEY not found. Some features may not work.")
            self.GEMINI_API_KEY = "demo-key"  # Fallback for demo
        
        if not self.DB_LINK:
            print(" Warning: DB_LINK not found. Database features disabled.")
            self.DB_LINK = None
        
        # Cấu hình MCP Client
        self.config = {
            "mcpServers": {
                "postgres": {
                    "command": "uv",
                "args": [
                    "run",
                    "postgres-mcp",
                    "--access-mode=unrestricted",
                ],
                    "env": {
                        "DATABASE_URI": self.DB_LINK
                    },
                }
            }
        }
        
        self.client = None
        self.agent = None
        self.llm = None
        
    async def _run_sql(self, sql: str) -> dict:
        session = self.client.get_session("postgres")
        try:
            print(f"🔧 Query Agent - Gọi execute_sql với: {sql}")
            resp = await session.call_tool("execute_sql", {"sql": sql})
            txt = resp.content[0].text if getattr(resp, "content", None) else str(resp)
            print(f"🔧 Query Agent - Kết quả execute_sql: {txt[:200]}...")
            
            # Parse kết quả từ database
            try:
                # Thử parse JSON trước
                import json
                obj = json.loads(txt)
                if isinstance(obj, list):
                    if obj:  # Có dữ liệu
                        # Lấy columns từ key đầu tiên
                        cols = list(obj[0].keys())
                        rows = [[row.get(c) for c in cols] for row in obj]
                        result = {"columns": cols, "data": rows}
                        print(f"✅ Query Agent - Parse JSON thành công: {result}")
                        return result
                    else:  # Empty list - trả về format đúng
                        # Tạo columns từ SQL SELECT
                        if "SELECT" in sql.upper():
                            select_part = sql.upper().split("SELECT")[1].split("FROM")[0].strip()
                            cols = [col.strip().split(" AS ")[-1].split(".")[-1] for col in select_part.split(",")]
                            result = {"columns": cols, "data": []}
                            print(f"✅ Query Agent - Empty result với columns: {result}")
                            return result
                        else:
                            result = {"columns": ["result"], "data": []}
                            print(f"✅ Query Agent - Empty result: {result}")
                            return result
                else:
                    print(f"⚠️ Query Agent - JSON không phải list: {obj}")
            except Exception as parse_error:
                print(f"⚠️ Query Agent - Không parse được JSON: {parse_error}")
                print(f"⚠️ Query Agent - Raw text: {txt}")
                
                # Thử parse với ast.literal_eval cho Python literal
                try:
                    import ast
                    # Clean up Decimal objects
                    cleaned_txt = txt.replace("Decimal('", "").replace("')", "")
                    obj = ast.literal_eval(cleaned_txt)
                    if isinstance(obj, list) and obj:
                        cols = list(obj[0].keys())
                        rows = [[row.get(c) for c in cols] for row in obj]
                        result = {"columns": cols, "data": rows}
                        print(f"✅ Query Agent - Parse ast.literal_eval thành công: {result}")
                        return result
                except Exception as ast_error:
                    print(f"⚠️ Query Agent - ast.literal_eval cũng lỗi: {ast_error}")
            
            # Fallback: trả về raw text
            return {"columns": ["result"], "data": [[txt]]}
            
        except Exception as e:
            print(f"❌ Query Agent - Lỗi execute_sql: {e}")
            # try list_objects/dry path
            return {"columns": ["result"], "data": [["Query failed"]]}

    async def get_table_details(self, table_name: str) -> dict:
        """Lấy thông tin chi tiết của một bảng cụ thể"""
        try:
            session = self.client.get_session("postgres")
            print(f"🔍 Query Agent - Lấy thông tin chi tiết bảng '{table_name}':")
            
            details = await session.call_tool("get_object_details", {
                "schema_name": "public",
                "object_name": table_name
            })
            
            if details and details.content:
                print(f"📋 Chi tiết bảng {table_name}:")
                print(details.content[0].text)
                return {
                    "success": True,
                    "table_name": table_name,
                    "details": details.content[0].text
                }
            else:
                return {
                    "success": False,
                    "error": f"Không thể lấy thông tin bảng {table_name}"
                }
                
        except Exception as e:
            print(f"❌ Lỗi khi lấy thông tin bảng {table_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _llm_generate_sql(self, natural_query: str, schema_hint: dict | None) -> str | None:
        """Dùng LLM sinh SQL từ yêu cầu tự nhiên. Không tạo biểu đồ, chỉ truy vấn."""
        try:
            tables = ", ".join(sorted(schema_hint.keys())) if schema_hint else "nhan_vien, phong_ban"
            # Sử dụng schema info nếu có
            schema_context = ""
            if hasattr(self, 'schema_info') and self.schema_info:
                schema_context = f"Cau truc bang: {self.schema_info}"
            
            prompt = (
                "Ban la chuyen gia SQL cho PostgreSQL. Nhiem vu duy nhat: tao mot cau SQL tra ve bang du lieu phu hop voi yeu cau. "
                "Tuyet doi khong giai thich, khong them loi, chi tra ve SQL hop le.\n\n"
                f"Yeu cau: {natural_query}\n"
                f"Cac bang co the dung (goi y): {tables}.\n"
                f"{schema_context}\n"
                "QUAN TRONG - TEN COT CHINH XAC (KHONG DUOC SAI):\n"
                "- nhan_vien: id, ho_ten (KHONG PHAI 'ten' hay 'nv.ten'), luong_co_ban, id_phong_ban, id_chuc_vu\n"
                "- phong_ban: id, ten_phong_ban (KHONG PHAI 'ten_phong'), ma_phong_ban\n"
                "- chuc_vu: id, ten_chuc_vu (KHONG PHAI 'ten_chuc'), ma_chuc_vu\n"
                "- cong_viec: id, ten_cong_viec, ma_cong_viec, id_du_an, id_nguoi_giao, id_nguoi_thuc_hien\n"
                "- du_an: id, ten_du_an (KHONG PHAI 'ten_du'), ma_du_an, id_phong_ban_chu_tri, id_quan_ly_du_an\n\n"
                "QUAN TRONG - JOIN RULES:\n"
                "1. Luon su dung table aliases: nv=nhan_vien, pb=phong_ban, cv=chuc_vu, c=cong_viec, da=du_an\n"
                "2. CHI JOIN khi co foreign key: nv.id_phong_ban = pb.id, nv.id_chuc_vu = cv.id\n"
                "3. Neu khong co foreign key, su dung CROSS JOIN hoac khong JOIN\n"
                "4. Kiem tra ten cot chinh xac truoc khi JOIN\n\n"
                "VI DU DUNG (COPY EXACT):\n"
                "SELECT nv.ho_ten FROM nhan_vien nv\n"
                "SELECT nv.ho_ten, pb.ten_phong_ban FROM nhan_vien nv JOIN phong_ban pb ON nv.id_phong_ban = pb.id\n"
                "SELECT cv.ten_chuc_vu, da.ten_du_an FROM chuc_vu cv, du_an da\n\n"
                "CHI tra ve SQL, KHONG boc trong markdown/code fence, KHONG them tien to ```sql."
            )
            sql_resp = await self.llm.ainvoke(prompt)
            sql_text = sql_resp.content if hasattr(sql_resp, 'content') else str(sql_resp)
            # Gỡ code-fence nếu LLM vẫn trả về ```sql ... ```
            text = sql_text.strip()
            if text.startswith("```"):
                # cắt phần trong khối fence đầu tiên
                start = text.find("```")
                end = text.rfind("```")
                inner = text[start+3:end] if end > start else text[start+3:]
                # loại bỏ nhãn ngôn ngữ 'sql' nếu có ở đầu
                inner = inner.lstrip().removeprefix("sql").lstrip()
                text = inner.strip()
            # Cố gắng lấy block chứa SELECT/WITH
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if any('select' in ln.lower() or 'with' in ln.lower() for ln in lines):
                return text
            # Validation: Kiểm tra SQL có đúng tên cột không
            if text:
                print(f"🔍 Query Agent - Validating SQL: {text}")
                if self._validate_sql_columns(text):
                    print(f"✅ Query Agent - SQL validation passed")
                    return text
                else:
                    print(f"❌ Query Agent - SQL validation failed: {text}")
                    return None
            else:
                print(f"⚠️ Query Agent - Empty SQL generated")
                return None
        except Exception:
            return None

    def _validate_sql_columns(self, sql: str) -> bool:
        """Kiểm tra SQL có sử dụng đúng tên cột không"""
        import re
        sql_lower = sql.lower()
        
        # Danh sách các tên cột SAI thường gặp
        wrong_columns = [
            'nv.ten', 'pb.ten_phong', 'cv.ten_chuc', 'da.ten_du',
            'c.id_chuc_vu', 'c.id_phong_ban', 'nv.luong', 'pb.ten_phong_ban'
        ]
        
        # Kiểm tra có tên cột sai không
        for wrong_col in wrong_columns:
            if wrong_col in sql_lower:
                print(f"❌ Query Agent - Found wrong column: {wrong_col}")
                return False
        
        # Extract all column references from SQL
        column_refs = re.findall(r'\b(\w+)\.(\w+)\b', sql_lower)
        print(f"🔍 Query Agent - Found column references: {column_refs}")
        
        # Validate each column reference against schema
        for table_alias, col_name in column_refs:
            if not self._is_valid_column(table_alias, col_name):
                print(f"❌ Query Agent - Invalid column: {table_alias}.{col_name}")
                return False
        
        return True
    
    def _is_valid_column(self, table_alias: str, col_name: str) -> bool:
        """Kiểm tra cột có tồn tại trong schema không"""
        # Map aliases to table names
        table_map = {
            'nv': 'nhan_vien', 'pb': 'phong_ban', 'cv': 'chuc_vu', 
            'c': 'cong_viec', 'da': 'du_an'
        }
        
        table_name = table_map.get(table_alias)
        if not table_name:
            return True  # Unknown alias, let it pass
        
        # Check if we have schema info for this table
        if table_name not in self.schema_details:
            print(f"⚠️ Query Agent - No schema info for {table_name}")
            return True  # No schema info, let it pass
        
        try:
            import json
            schema = json.loads(self.schema_details[table_name])
            columns = [col['column'] for col in schema.get('columns', [])]
            
            is_valid = col_name in columns
            if not is_valid:
                print(f"❌ Query Agent - Column {col_name} not found in {table_name}. Available: {columns}")
            
            return is_valid
        except Exception as e:
            print(f"⚠️ Query Agent - Error checking column {col_name} in {table_name}: {e}")
            return True  # Error in validation, let it pass

    def _fallback_sql(self, question: str) -> str | None:
        q = question.lower()
        # heuristics for common intents
        if ("so sánh" in q or "so sanh" in q or "biểu đồ" in q or "bieu do" in q) and ("phòng ban" in q or "phong ban" in q):
            return (
                "SELECT pb.ten_phong_ban AS phong_ban, COUNT(nv.id) AS so_nhan_vien "
                "FROM phong_ban pb LEFT JOIN nhan_vien nv ON pb.id = nv.id_phong_ban "
                "GROUP BY pb.ten_phong_ban ORDER BY so_nhan_vien DESC;"
            )
        if ("lương" in q or "luong" in q) and ("trung bình" in q or "trung binh" in q) and ("phòng ban" in q or "phong ban" in q):
            return (
                "SELECT pb.ten_phong_ban AS phong_ban, ROUND(AVG(nv.luong_co_ban),2) AS luong_trung_binh "
                "FROM nhan_vien nv JOIN phong_ban pb ON nv.id_phong_ban = pb.id "
                "GROUP BY pb.ten_phong_ban ORDER BY luong_trung_binh DESC;"
            )
        if ("lương" in q or "luong" in q) and ("thấp nhất" in q or "thap nhat" in q):
            return (
                "SELECT ho_ten, luong_co_ban FROM nhan_vien "
                "WHERE luong_co_ban IS NOT NULL "
                "ORDER BY luong_co_ban ASC LIMIT 1;"
            )
        if ("lương" in q or "luong" in q) and ("cao nhất" in q or "cao nhat" in q):
            return (
                "SELECT ho_ten, luong_co_ban FROM nhan_vien "
                "WHERE luong_co_ban IS NOT NULL "
                "ORDER BY luong_co_ban DESC LIMIT 1;"
            )
        if ("ngẫu nhiên" in q or "ngau nhien" in q) and ("nhân viên" in q or "nhan vien" in q):
            return (
                "SELECT nv.id, nv.ho_ten, cv.ten_chuc_vu, pb.ten_phong_ban, nv.luong_co_ban "
                "FROM nhan_vien nv JOIN chuc_vu cv ON nv.id_chuc_vu = cv.id JOIN phong_ban pb ON nv.id_phong_ban = pb.id "
                "ORDER BY RANDOM() LIMIT 5;"
            )
        if ("công việc" in q or "cong viec" in q) and ("danh sách" in q or "danh sach" in q):
            return (
                "SELECT cv.id, cv.ten_cong_viec, cv.trang_thai, cv.do_uu_tien "
                "FROM cong_viec cv ORDER BY cv.created_at DESC LIMIT 10;"
            )
        if ("tên" in q or "ten" in q) and ("nhân viên" in q or "nhan vien" in q):
            return (
                "SELECT ho_ten FROM nhan_vien ORDER BY ho_ten LIMIT 10;"
            )
        if ("nhân viên" in q or "nhan vien" in q) and ("danh sách" in q or "danh sach" in q):
            return (
                "SELECT id, ho_ten, luong_co_ban FROM nhan_vien ORDER BY ho_ten LIMIT 10;"
            )
        if ("dự án" in q or "du an" in q) and ("danh sách" in q or "danh sach" in q):
            return (
                "SELECT id, ten_du_an, trang_thai, ngan_sach FROM du_an ORDER BY created_at DESC LIMIT 10;"
            )
        if ("dự án" in q or "du an" in q) and ("hiện" in q or "hien" in q):
            return (
                "SELECT ten_du_an, trang_thai, ngan_sach FROM du_an WHERE trang_thai != 'hoan_thanh' ORDER BY created_at DESC;"
            )
        return None

    def _summarize_table(self, question: str, table: dict) -> str:
        try:
            cols = [str(c).lower() for c in table.get("columns", [])]
            data = table.get("data", [])
            if not cols or not data:
                return ""
            # Lương thấp nhất
            if any("luong" in c for c in cols) and any("ho_ten" == c or "ten" in c for c in cols) and len(data) >= 1:
                try:
                    name_idx = next(i for i,c in enumerate(cols) if c == "ho_ten" or "ten" in c)
                    sal_idx = next(i for i,c in enumerate(cols) if "luong" in c)
                    name = data[0][name_idx]
                    sal = data[0][sal_idx]
                    return f"Nhân viên có lương thấp nhất là {name} với mức lương {sal}."
                except Exception:
                    pass
            # Số lượng nhân viên theo phòng ban
            if ("phong_ban" in cols or "phongban" in cols or "phong ban" in question.lower()) and any("so_luong" in c or "so_nhan_vien" in c for c in cols):
                try:
                    dept_idx = next(i for i,c in enumerate(cols) if c in ("phong_ban","phongban","phongban","phong ban","ten_phong_ban"))
                    cnt_idx = next(i for i,c in enumerate(cols) if "so_luong" in c or "so_nhan_vien" in c)
                    top = data[:3]
                    parts = [f"{row[dept_idx]}: {row[cnt_idx]}" for row in top]
                    return "Số lượng nhân viên theo phòng ban (top 3): " + ", ".join(parts) + "."
                except Exception:
                    pass
            # Lương trung bình theo phòng ban
            if any("trung_binh" in c for c in cols) and ("phong_ban" in cols or "ten_phong_ban" in cols):
                try:
                    dept_idx = next(i for i,c in enumerate(cols) if c in ("phong_ban","ten_phong_ban"))
                    avg_idx = next(i for i,c in enumerate(cols) if "trung_binh" in c)
                    top = data[:3]
                    parts = [f"{row[dept_idx]}: {row[avg_idx]}" for row in top]
                    return "Lương trung bình theo phòng ban (top 3): " + ", ".join(parts) + "."
                except Exception:
                    pass
            # Mặc định: thông báo số dòng
            return f"Đã truy vấn thành công {len(data)} dòng dữ liệu."
        except Exception:
            return ""

    async def _load_schema_details(self):
        """Lấy thông tin chi tiết các bảng quan trọng"""
        if not self.client:
            return
            
        print(" Query Agent: Dang lay thong tin chi tiet cac bang...")
        session = self.client.get_session("postgres")
        important_tables = ['nhan_vien', 'phong_ban', 'chuc_vu', 'cong_viec', 'du_an']
        
        self.schema_details = {}
        for table_name in important_tables:
            try:
                details = await session.call_tool("get_object_details", {
                    "schema_name": "public",
                    "object_name": table_name
                })
                if details and details.content:
                    self.schema_details[table_name] = details.content[0].text
                    print(f" Query Agent: Da lay thong tin bang {table_name}")
            except Exception as e:
                print(f" Query Agent: Khong the lay thong tin bang {table_name}: {e}")
        
        # Tạo schema info string cho prompt
        self.schema_info = ""
        print(f"🔍 Query Agent - Building schema info from {len(self.schema_details)} tables")
        
        for table_name, details in self.schema_details.items():
            try:
                import json
                detail_obj = json.loads(details)
                if 'columns' in detail_obj:
                    columns = [col['column'] for col in detail_obj['columns']]
                    self.schema_info += f"`{table_name}` co cac cot: {', '.join(columns)}. "
                    print(f"✅ Query Agent - Schema for {table_name}: {columns}")
            except Exception as e:
                print(f"❌ Query Agent - Error parsing schema for {table_name}: {e}")
                self.schema_info += f"`{table_name}` (thong tin chi tiet co san). "
        
        print(f"🔍 Query Agent - Final schema info: {self.schema_info[:200]}...")

    async def initialize(self):
        """Khởi tạo MCP Client và Agent với Schema Awareness"""
        if self.client is None:
            try:
                print(" Query Agent: Dang khoi tao MCP Client...")
                
                # Kiểm tra MCPClient có method from_dict không
                if hasattr(MCPClient, 'from_dict'):
                    print(" Query Agent: Su dung MCPClient.from_dict")
                    self.client = MCPClient.from_dict(self.config)
                else:
                    # Fallback: tạo client trực tiếp
                    print(" Query Agent: MCPClient.from_dict khong kha dung, su dung fallback")
                    try:
                        # Thử tạo client với config trực tiếp
                        self.client = MCPClient()
                        # Set config manually nếu có method
                        if hasattr(self.client, 'configure'):
                            self.client.configure(self.config)
                    except Exception as e:
                        print(f" Query Agent: Không thể tạo MCPClient: {e}")
                        raise e
                
                try:
                    await self.client.create_all_sessions()
                    print(" Query Agent: MCP Client da ket noi!")
                    
                    # Lấy thông tin schema chi tiết
                    await self._load_schema_details()
                    
                except Exception as session_error:
                    print(f" Query Agent: Loi tao sessions: {session_error}")
                    # Thử tạo session riêng lẻ
                    try:
                        await self.client.create_session("postgres")
                        print(" Query Agent: PostgreSQL session da tao!")
                    except Exception as postgres_error:
                        print(f" Query Agent: Không thể tạo PostgreSQL session: {postgres_error}")
                        raise postgres_error
                
                # Đợi MCP Server load schema
                await asyncio.sleep(3)
                
                # Khởi tạo LLM
                self.llm = ChatGoogleGenerativeAI(
                    model="models/gemini-2.5-flash-lite",
                    google_api_key=self.GEMINI_API_KEY,
                    temperature=0.2,
                )
                
                # Tạo MCP Agent
                try:
                    self.agent = MCPAgent(llm=self.llm, client=self.client, max_steps=20)
                    print(" Query Agent: Sẵn sàng xử lý truy vấn!")
                except Exception as agent_error:
                    print(f" Query Agent: Lỗi tạo MCPAgent: {agent_error}")
                    # Agent có thể không cần thiết nếu có LLM và client
                    self.agent = None
                    print(" Query Agent: Chạy ở chế độ không có MCPAgent")
                
            except Exception as e:
                print(f" Query Agent: Loi khoi tao MCP Client: {e}")
                print(" Query Agent: Chuyển sang mock mode")
                # Khởi tạo LLM cho mock mode
                self.llm = ChatGoogleGenerativeAI(
                    model="models/gemini-2.5-flash-lite",
                    google_api_key=self.GEMINI_API_KEY,
                    temperature=0.2,
                )
                self.client = None
                self.agent = None
    
    async def get_schema_info(self):
        """Lấy thông tin schema để agent hiểu cấu trúc database"""
        try:
            session = self.client.get_session("postgres")
            
            # Liệt kê tools
            tools = await session.list_tools()
            print("🧰 Query Agent - Tools khả dụng:")
            for t in tools:
                print(f" - {t.name}: {t.description}")
            
            # Lấy schema và bảng
            schema_map = {}
            try:
                schemas = await session.call_tool("list_schemas", {})
                if schemas and schemas.content:
                    print(" Query Agent - Schema có sẵn:")
                    print(schemas.content[0].text)
                    
                    # Lấy bảng trong schema 'public'
                    tables = await session.call_tool("list_objects", {"schema_name": "public"})
                    if tables and tables.content:
                        text = tables.content[0].text
                        print("\n Query Agent - Bảng có trong schema 'public':")
                        print(text)
                        
                        # Lưu vào schema_map và lấy thông tin chi tiết bảng
                        table_names = []
                        print(f"🔍 Query Agent - Raw schema text: {text}")
                        
                        # Parse JSON từ text
                        try:
                            import json
                            schema_data = json.loads(text)
                            if isinstance(schema_data, list):
                                for item in schema_data:
                                    if isinstance(item, dict) and 'name' in item:
                                        name = item['name']
                                        schema_map[name.lower()] = "public"
                                        table_names.append(name)
                                        print(f"✅ Query Agent - Tìm thấy bảng: {name}")
                        except Exception as json_error:
                            print(f"⚠️ Query Agent - Không parse được JSON, thử parse text: {json_error}")
                            # Fallback: parse text - sửa để lấy tất cả bảng
                            lines = text.splitlines()
                            print(f"🔍 Query Agent - Số dòng text: {len(lines)}")
                            for i, line in enumerate(lines):
                                print(f"🔍 Query Agent - Dòng {i}: {line}")
                                if "'name':" in line:
                                    try:
                                        # Tìm tất cả 'name': trong dòng
                                        parts = line.split("'name':")
                                        for part in parts[1:]:  # Bỏ qua phần đầu
                                            name = part.split("'")[1]
                                            schema_map[name.lower()] = "public"
                                            table_names.append(name)
                                            print(f"✅ Query Agent - Tìm thấy bảng (text): {name}")
                                    except Exception as parse_error:
                                        print(f"❌ Query Agent - Lỗi parse line: {line}, error: {parse_error}")
                        
                        # Tự động lấy thông tin chi tiết cho các bảng quan trọng
                        important_tables = ['cong_viec', 'nhan_vien', 'phong_ban', 'chuc_vu', 'du_an']
                        print(f"🔍 Query Agent - Bảng tìm thấy: {table_names}")
                        print(f"🔍 Query Agent - Bảng quan trọng: {important_tables}")
                        
                        for table_name in important_tables:
                            if table_name in table_names:
                                try:
                                    print(f"\n🔍 Query Agent - Lấy thông tin chi tiết bảng '{table_name}':")
                                    details = await session.call_tool("get_object_details", {
                                        "schema_name": "public",
                                        "object_name": table_name
                                    })
                                    if details and details.content:
                                        print(f"📋 Chi tiết bảng {table_name}:")
                                        print(details.content[0].text)
                                    else:
                                        print(f"⚠️ Không có nội dung chi tiết cho bảng {table_name}")
                                except Exception as e:
                                    print(f"❌ Không thể lấy chi tiết bảng {table_name}: {e}")
                            else:
                                print(f"⚠️ Bảng {table_name} không tìm thấy trong danh sách")
            except Exception as e:
                print(f" Query Agent - Không thể lấy danh sách schema: {e}")
            
            return schema_map
            
        except Exception as e:
            print(f" Query Agent - Lỗi khi lấy schema: {e}")
            return {}
    
    async def process(self, user_input: str) -> dict:
        """
        Xử lý truy vấn của người dùng
        """
        print(f" Query Agent: Bat dau process voi input: '{user_input}'")
        try:
            # Khởi tạo nếu chưa có
            await self.initialize()
            
            print(f" Query Agent: Xu ly truy van '{user_input}'")
            
            # Lấy thông tin schema
            schema_map = await self.get_schema_info()
            
            # Tạo hint cho LLM
            hint = (
                "NHIỆM VỤ DUY NHẤT: Truy vấn dữ liệu và trả về BẢNG có {columns, data}. "
                "TUYỆT ĐỐI KHÔNG nói về vẽ biểu đồ, không trả lời rằng không thể vẽ biểu đồ, không yêu cầu thêm thông tin. "
                "Nếu thiếu ngữ cảnh, tự suy luận và ưu tiên bảng trong schema 'public' như 'nhan_vien', 'phong_ban'. "
                "Chỉ trả về dữ liệu bảng hoặc JSON list[object]."
            )
            
            # Debug: Kiểm tra schema trước khi generate SQL
            print(f"🔍 Query Agent - Schema map: {schema_map}")
            print(f"🔍 Query Agent - Schema details available: {list(self.schema_details.keys())}")
            
            # ƯU TIÊN: LLM → SQL dựa trên schema, rồi chạy trực tiếp
            sql_direct = await self._llm_generate_sql(user_input, schema_map)
            print(f"🔍 Query Agent - SQL được tạo: {sql_direct}")
            
            if sql_direct:
                print(f"🚀 Query Agent - Thực thi SQL: {sql_direct}")
                table = await self._run_sql(sql_direct)
                print(f"📊 Query Agent - Kết quả SQL: {table}")
                final_answer = self._summarize_table(user_input, table)
                return {
                    "agent": "query_agent",
                    "status": "success",
                    "result": table,
                    "final_answer": final_answer,
                    "raw_result": sql_direct
                }
            else:
                print("❌ Query Agent - Không thể tạo SQL, thử fallback")
                # Thử fallback SQL
                fallback_sql = self._fallback_sql(user_input)
                if fallback_sql:
                    print(f"🔄 Query Agent - Sử dụng fallback SQL: {fallback_sql}")
                    table = await self._run_sql(fallback_sql)
                    print(f"📊 Query Agent - Kết quả fallback: {table}")
                    final_answer = self._summarize_table(user_input, table)
                    return {
                        "agent": "query_agent",
                        "status": "success",
                        "result": table,
                        "final_answer": final_answer,
                        "raw_result": fallback_sql
                }

            # Fallback: dùng toolflow của MCP agent (nếu có)
            if self.agent:
                result = await self.agent.run(hint + user_input)
            else:
                # Nếu không có agent, trả về error
                return {
                    "agent": "query_agent",
                    "status": "error",
                    "error": "MCP Agent not available - database connection failed",
                    "result": {"columns": ["error"], "data": [["Database not available"]]}
                }

            # ÉP LUÔN trả về kết quả có cấu trúc {columns, data}
            def safe(val):
                """Convert value to JSON-safe format"""
                if val is None:
                    return ""
                if isinstance(val, (int, float)):
                    return val
                return str(val)
            
            def to_table(payload: str) -> dict:
                try:
                    text = payload.strip()
                    # Remove markdown code fences if present
                    if text.startswith("```"):
                        lines = text.splitlines()
                        # drop first fence line and possible language tag
                        if lines and lines[0].startswith("```"):
                            lines = lines[1:]
                        # drop trailing fence line
                        if lines and lines[-1].strip().startswith("```"):
                            lines = lines[:-1]
                        text = "\n".join(lines).strip()
                    # Try JSON first
                    obj = json.loads(text)
                    # Nếu đã có columns/data thì dùng luôn
                    if isinstance(obj, dict) and "columns" in obj and "data" in obj:
                        return obj
                    # Nếu là list[dict] → chuẩn hoá
                    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
                        cols = list(obj[0].keys()) if obj else []
                        rows = [[safe(row.get(c)) for c in cols] for row in obj]
                        return {"columns": cols, "data": rows}
                except Exception:
                    # Not a JSON; try Python literal list of dicts
                    try:
                        import ast
                        lit = ast.literal_eval(text)
                        if isinstance(lit, list) and (len(lit) == 0 or isinstance(lit[0], dict)):
                            cols = list(lit[0].keys()) if lit else []
                            rows = [[safe(row.get(c)) for c in cols] for row in lit]
                            return {"columns": cols, "data": rows}
                        if isinstance(lit, dict) and "columns" in lit and "data" in lit:
                            cols = list(lit["columns"])
                            data_rows = [[safe(v) for v in row] for row in list(lit["data"]) ]
                            return {"columns": cols, "data": data_rows}
                    except Exception:
                        pass
                # Fallback: cố gắng tách bảng markdown đơn giản → 2 cột
                try:
                    lines = [ln.strip() for ln in payload.splitlines() if "|" in ln]
                    if len(lines) >= 2:
                        headers = [h.strip() for h in lines[0].split("|") if h.strip()]
                        rows = []
                        for ln in lines[1:]:
                            cells = [c.strip() for c in ln.split("|") if c.strip()]
                            if len(cells) == len(headers):
                                rows.append(cells)
                        if rows:
                            return {"columns": headers, "data": rows}
                except Exception:
                    pass
                # Thử parse chuỗi JSON chứa list-dict (trường hợp từ MCP agent)
                try:
                    # Tìm pattern [{'key': 'value'}, ...] trong text
                    json_pattern = r'\[.*?\]'
                    matches = re.findall(json_pattern, payload, re.DOTALL)
                    for match in matches:
                        try:
                            # Thử parse trực tiếp với ast.literal_eval (Python syntax)
                            import ast
                            parsed = ast.literal_eval(match)
                            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                                cols = list(parsed[0].keys())
                                rows = [[safe(row.get(c)) for c in cols] for row in parsed]
                                return {"columns": cols, "data": rows}
                        except:
                            # Thử JSON parse
                            try:
                                parsed = json.loads(match)
                                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                                    cols = list(parsed[0].keys())
                                    rows = [[safe(row.get(c)) for c in cols] for row in parsed]
                                    return {"columns": cols, "data": rows}
                            except:
                                continue
                except:
                    pass
                
                # Cuối cùng: trả text vào 1 cột
                return {"columns": ["result"], "data": [[payload]]}
            
            # Parse kết quả và chuẩn hóa cho Chart Agent
            try:
                print(f" Query Agent: Bắt đầu parse kết quả - type: {type(result)}")
                print(f" Query Agent: Kết quả content: {str(result)[:200]}...")
                
                if isinstance(result, str):
                    # Nếu kết quả là hướng dẫn/không hữu ích → cưỡng bức query trực tiếp
                    guidance_markers = [
                        "Tôi không thể vẽ biểu đồ", "Tôi cần thêm thông tin", "list_schemas", "get_object_details", "list_objects",
                        "I cannot draw", "I need more information"
                    ]
                    if any(m in result for m in guidance_markers):
                        print(f" Query Agent: Phát hiện guidance markers, sử dụng fallback SQL")
                        sql = self._fallback_sql(user_input)
                        if sql:
                            print(f" Query Agent: Fallback SQL: {sql}")
                            table = await self._run_sql(sql)
                        else:
                            print(f" Query Agent: Không có fallback SQL, sử dụng to_table")
                            table = to_table(result)
                    else:
                        print(f" Query Agent: Không phải guidance, sử dụng to_table")
                        table = to_table(result)
                    
                    # CƯỠNG BỨC PARSE: Nếu table chỉ có 1 cột "result" chứa JSON string
                    print(f" Query Agent: Kiểm tra table format - columns: {table.get('columns')}, data length: {len(table.get('data', []))}")
                    if (table.get("columns") == ["result"] and 
                        table.get("data") and 
                        len(table["data"]) == 1 and 
                        isinstance(table["data"][0][0], str) and 
                        table["data"][0][0].strip().startswith('[')):
                        
                        print(f" Query Agent: Phát hiện JSON string trong cột result, bắt đầu parse...")
                        # Thử parse JSON string trong cột result
                        json_str = table["data"][0][0]
                        print(f" Query Agent: JSON string để parse: {json_str[:100]}...")
                        try:
                            import ast
                            # Loại bỏ Decimal('...') để có thể parse
                            json_clean = re.sub(r"Decimal\('([^']+)'\)", r"\1", json_str)
                            print(f" Query Agent: JSON sau khi clean: {json_clean[:100]}...")
                            parsed = ast.literal_eval(json_clean)
                            print(f" Query Agent: Parsed result type: {type(parsed)}, length: {len(parsed) if isinstance(parsed, list) else 'N/A'}")
                            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                                cols = list(parsed[0].keys())
                                rows = [[safe(row.get(c)) for c in cols] for row in parsed]
                                table = {"columns": cols, "data": rows}
                                print(f" Query Agent: Đã parse thành công JSON string thành bảng {len(cols)} cột, {len(rows)} dòng")
                                print(f" Query Agent: Columns: {cols}")
                                print(f" Query Agent: Sample data: {rows[:2] if rows else 'No data'}")
                            else:
                                print(f" Query Agent: Parsed data không phải list[dict]: {type(parsed)}")
                        except Exception as e:
                            print(f" Query Agent: Không thể parse JSON string: {e}")
                            import traceback
                            print(f" Query Agent: Traceback: {traceback.format_exc()}")
                    else:
                        print(f" Query Agent: Không thỏa mãn điều kiện parse - columns: {table.get('columns')}, data: {table.get('data')}")
                    
                    # Nếu agent không trả data → cố gắng cưỡng bức thực thi SQL rõ ràng
                    if not table.get("data") or (len(table.get("columns", [])) == 1 and table["columns"][0] == "result"):
                        # Thử LLM -> SQL
                        sql = await self._llm_generate_sql(user_input, schema_map)
                        if not sql:
                            sql = self._fallback_sql(user_input)
                        if sql:
                            table = await self._run_sql(sql)
                    final_answer = self._summarize_table(user_input, table)
                    return {
                        "agent": "query_agent",
                        "status": "success",
                        "result": table,
                        "final_answer": final_answer,
                        "raw_result": result
                    }
                else:
                    # Nếu SDK trả object → chuyển thành bảng
                    print(f" Query Agent: Xử lý object result - type: {type(result)}")
                    result_str = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                    print(f" Query Agent: Object result string: {result_str[:200]}...")
                    table = to_table(result_str)
                    print(f" Query Agent: Sau to_table - columns: {table.get('columns')}, data: {table.get('data')}")
                    
                    # CƯỠNG BỨC PARSE: Nếu table chỉ có 1 cột "result" chứa JSON string
                    print(f" Query Agent (object): Kiểm tra table format - columns: {table.get('columns')}, data length: {len(table.get('data', []))}")
                    if (table.get("columns") == ["result"] and 
                        table.get("data") and 
                        len(table["data"]) == 1 and 
                        isinstance(table["data"][0][0], str) and 
                        table["data"][0][0].strip().startswith('[')):
                        
                        print(f" Query Agent (object): Phát hiện JSON string trong cột result, bắt đầu parse...")
                        # Thử parse JSON string trong cột result
                        json_str = table["data"][0][0]
                        print(f" Query Agent (object): JSON string để parse: {json_str[:100]}...")
                        try:
                            import ast
                            # Loại bỏ Decimal('...') để có thể parse
                            json_clean = re.sub(r"Decimal\('([^']+)'\)", r"\1", json_str)
                            print(f" Query Agent (object): JSON sau khi clean: {json_clean[:100]}...")
                            parsed = ast.literal_eval(json_clean)
                            print(f" Query Agent (object): Parsed result type: {type(parsed)}, length: {len(parsed) if isinstance(parsed, list) else 'N/A'}")
                            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                                cols = list(parsed[0].keys())
                                rows = [[safe(row.get(c)) for c in cols] for row in parsed]
                                table = {"columns": cols, "data": rows}
                                print(f" Query Agent (object): Đã parse thành công JSON string thành bảng {len(cols)} cột, {len(rows)} dòng")
                                print(f" Query Agent (object): Columns: {cols}")
                                print(f" Query Agent (object): Sample data: {rows[:2] if rows else 'No data'}")
                            else:
                                print(f" Query Agent (object): Parsed data không phải list[dict]: {type(parsed)}")
                        except Exception as e:
                            print(f" Query Agent (object): Không thể parse JSON string: {e}")
                            import traceback
                            print(f" Query Agent (object): Traceback: {traceback.format_exc()}")
                    else:
                        print(f" Query Agent (object): Không thỏa mãn điều kiện parse - columns: {table.get('columns')}, data: {table.get('data')}")
                    
                    if not table.get("data") or (len(table.get("columns", [])) == 1 and table["columns"][0] == "result"):
                        sql = await self._llm_generate_sql(user_input, schema_map)
                        if not sql:
                            sql = self._fallback_sql(user_input)
                        if sql:
                            table = await self._run_sql(sql)
                    final_answer = self._summarize_table(user_input, table)
                    return {
                        "agent": "query_agent",
                        "status": "success", 
                        "result": table,
                        "final_answer": final_answer,
                        "raw_result": result_str
                    }
            except Exception as parse_error:
                return {
                    "agent": "query_agent",
                    "status": "success",
                    "result": {"text": str(result)},
                    "raw_result": str(result),
                    "parse_warning": f"Không thể parse kết quả: {parse_error}"
                }
                
        except Exception as e:
            return {
                "agent": "query_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def test_connection(self):
        """Test MCP connection và database"""
        try:
            print("=== TESTING QUERY AGENT CONNECTION ===")
            
            # Test MCP connection
            print("1. Testing MCP connection...")
            if not self.client:
                await self.initialize()
            
            if not self.client:
                print("   [ERROR] MCP Client not available")
                return False
            
            print("   [OK] MCP Client connected")
            
            # Test PostgreSQL session
            print("2. Testing PostgreSQL session...")
            try:
                postgres_session = self.client.get_session("postgres")
                print("   [OK] PostgreSQL session available")
                
                # Test tools
                print("3. Testing MCP tools...")
                tools = await postgres_session.list_tools()
                print(f"   [OK] Available tools: {len(tools)}")
                for tool in tools[:3]:  # Show first 3 tools
                    print(f"     - {tool.name}: {tool.description[:50]}...")
                
            except Exception as e:
                print(f"   [ERROR] PostgreSQL session failed: {e}")
                return False
            
            # Test database query
            print("4. Testing database query...")
            try:
                test_queries = [
                    "Tong so nhan vien",
                    "Danh sach nhan vien", 
                    "Luong trung binh"
                ]
                
                for i, query in enumerate(test_queries):
                    print(f"   Test {i+1}: '{query}'")
                    result = await self.process(query)
                    
                    if result.get('status') == 'success':
                        print(f"   [OK] Query successful")
                        print(f"   [OK] Result type: {type(result.get('result'))}")
                    else:
                        print(f"   [ERROR] Query failed: {result.get('error')}")
                        return False
                
            except Exception as e:
                print(f"   [ERROR] Database query failed: {e}")
                return False
            
            print("\n=== QUERY AGENT CONNECTION TEST PASSED ===")
            return True
            
        except Exception as e:
            print(f"[ERROR] CONNECTION TEST FAILED: {e}")
            return False

    async def close(self):
        """Đóng kết nối"""
        if self.client:
            await self.client.close_all_sessions()
            print("🔌 Query Agent: Đã đóng kết nối MCP")

# Test function
async def test_query_agent():
    agent = QueryAgent()
    
    test_queries = [
        "Liệt kê tất cả nhân viên",
        "Tìm nhân viên có lương cao nhất",
        "Thống kê nhân viên theo phòng ban",
        "Hiển thị thông tin chi tiết của bảng NhanVien"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Test Query: {query}")
        result = await agent.process(query)
        print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    await agent.close()

if __name__ == "__main__":
    asyncio.run(test_query_agent())
