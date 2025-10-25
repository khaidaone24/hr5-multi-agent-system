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
        
        if not self.DB_LINK or not self.GEMINI_API_KEY:
            raise ValueError("⚠️ Thiếu DB_LINK hoặc GOOGLE_API_KEY trong .env")
        
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
            resp = await session.call_tool("execute_sql", {"sql": sql})
            txt = resp.content[0].text if getattr(resp, "content", None) else str(resp)
        except Exception:
            # try list_objects/dry path
            return {"columns": ["result"], "data": [["Query failed"]]}

        # normalize
        try:
            obj = json.loads(txt)
            if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
                cols = list(obj[0].keys()) if obj else []
                rows = [[row.get(c) for c in cols] for row in obj]
                return {"columns": cols, "data": rows}
            if isinstance(obj, dict) and "columns" in obj and "data" in obj:
                return obj
        except Exception:
            pass
        # fallback single column
        return {"columns": ["result"], "data": [[txt]]}

    async def _llm_generate_sql(self, natural_query: str, schema_hint: dict | None) -> str | None:
        """Dùng LLM sinh SQL từ yêu cầu tự nhiên. Không tạo biểu đồ, chỉ truy vấn."""
        try:
            tables = ", ".join(sorted(schema_hint.keys())) if schema_hint else "nhan_vien, phong_ban"
            prompt = (
                "Bạn là chuyên gia SQL cho PostgreSQL. Nhiệm vụ duy nhất: tạo một câu SQL trả về bảng dữ liệu phù hợp với yêu cầu. "
                "Tuyệt đối không giải thích, không thêm lời, chỉ trả về SQL hợp lệ.\n\n"
                f"Yêu cầu: {natural_query}\n"
                f"Các bảng có thể dùng (gợi ý): {tables}.\n"
                "Một số cột phổ biến có thể có: nhan_vien(id, ho_ten, id_phong_ban, luong_co_ban), phong_ban(id, ten_phong_ban).\n"
                "Ví dụ: lương trung bình theo phòng ban, đếm số nhân viên theo phòng ban...\n"
                "CHỈ trả về SQL, KHÔNG bọc trong markdown/code fence, KHÔNG thêm tiền tố ```sql."
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
            return text or None
        except Exception:
            return None

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
                "ORDER BY luong_co_ban ASC LIMIT 1;"
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

    async def initialize(self):
        """Khởi tạo MCP Client và Agent"""
        if self.client is None:
            print("🚀 Query Agent: Đang khởi tạo MCP Client...")
            self.client = MCPClient.from_dict(self.config)
            await self.client.create_all_sessions()
            print("✅ Query Agent: MCP Client đã kết nối!")
            
            # Đợi MCP Server load schema
            await asyncio.sleep(3)
            
            # Khởi tạo LLM
            self.llm = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash-lite",
                google_api_key=self.GEMINI_API_KEY,
                temperature=0.2,
            )
            
            # Tạo MCP Agent
            self.agent = MCPAgent(llm=self.llm, client=self.client, max_steps=20)
            print("🤖 Query Agent: Sẵn sàng xử lý truy vấn!")
    
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
                    print("✅ Query Agent - Schema có sẵn:")
                    print(schemas.content[0].text)
                    
                    # Lấy bảng trong schema 'public'
                    tables = await session.call_tool("list_objects", {"schema_name": "public"})
                    if tables and tables.content:
                        text = tables.content[0].text
                        print("\n📊 Query Agent - Bảng có trong schema 'public':")
                        print(text)
                        
                        # Lưu vào schema_map
                        for line in text.splitlines():
                            if "'name':" in line:
                                name = line.split("'name':")[1].split("'")[1]
                                schema_map[name.lower()] = "public"
            except Exception as e:
                print(f"⚠️ Query Agent - Không thể lấy danh sách schema: {e}")
            
            return schema_map
            
        except Exception as e:
            print(f"❌ Query Agent - Lỗi khi lấy schema: {e}")
            return {}
    
    async def process(self, user_input: str) -> dict:
        """
        Xử lý truy vấn của người dùng
        """
        print(f"🔍 Query Agent: Bắt đầu process với input: '{user_input}'")
        try:
            # Khởi tạo nếu chưa có
            await self.initialize()
            
            print(f"🔍 Query Agent: Xử lý truy vấn '{user_input}'")
            
            # Lấy thông tin schema
            schema_map = await self.get_schema_info()
            
            # Tạo hint cho LLM
            hint = (
                "NHIỆM VỤ DUY NHẤT: Truy vấn dữ liệu và trả về BẢNG có {columns, data}. "
                "TUYỆT ĐỐI KHÔNG nói về vẽ biểu đồ, không trả lời rằng không thể vẽ biểu đồ, không yêu cầu thêm thông tin. "
                "Nếu thiếu ngữ cảnh, tự suy luận và ưu tiên bảng trong schema 'public' như 'nhan_vien', 'phong_ban'. "
                "Chỉ trả về dữ liệu bảng hoặc JSON list[object]."
            )
            
            # ƯU TIÊN: LLM → SQL dựa trên schema, rồi chạy trực tiếp
            sql_direct = await self._llm_generate_sql(user_input, schema_map)
            if sql_direct:
                table = await self._run_sql(sql_direct)
                final_answer = self._summarize_table(user_input, table)
                return {
                    "agent": "query_agent",
                    "status": "success",
                    "result": table,
                    "final_answer": final_answer,
                    "raw_result": sql_direct
                }

            # Fallback: dùng toolflow của MCP agent
            result = await self.agent.run(hint + user_input)

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
                print(f"🔍 Query Agent: Bắt đầu parse kết quả - type: {type(result)}")
                print(f"🔍 Query Agent: Kết quả content: {str(result)[:200]}...")
                
                if isinstance(result, str):
                    # Nếu kết quả là hướng dẫn/không hữu ích → cưỡng bức query trực tiếp
                    guidance_markers = [
                        "Tôi không thể vẽ biểu đồ", "Tôi cần thêm thông tin", "list_schemas", "get_object_details", "list_objects",
                        "I cannot draw", "I need more information"
                    ]
                    if any(m in result for m in guidance_markers):
                        print(f"🔍 Query Agent: Phát hiện guidance markers, sử dụng fallback SQL")
                        sql = self._fallback_sql(user_input)
                        if sql:
                            print(f"🔍 Query Agent: Fallback SQL: {sql}")
                            table = await self._run_sql(sql)
                        else:
                            print(f"🔍 Query Agent: Không có fallback SQL, sử dụng to_table")
                            table = to_table(result)
                    else:
                        print(f"🔍 Query Agent: Không phải guidance, sử dụng to_table")
                        table = to_table(result)
                    
                    # CƯỠNG BỨC PARSE: Nếu table chỉ có 1 cột "result" chứa JSON string
                    print(f"🔍 Query Agent: Kiểm tra table format - columns: {table.get('columns')}, data length: {len(table.get('data', []))}")
                    if (table.get("columns") == ["result"] and 
                        table.get("data") and 
                        len(table["data"]) == 1 and 
                        isinstance(table["data"][0][0], str) and 
                        table["data"][0][0].strip().startswith('[')):
                        
                        print(f"🔍 Query Agent: Phát hiện JSON string trong cột result, bắt đầu parse...")
                        # Thử parse JSON string trong cột result
                        json_str = table["data"][0][0]
                        print(f"🔍 Query Agent: JSON string để parse: {json_str[:100]}...")
                        try:
                            import ast
                            # Loại bỏ Decimal('...') để có thể parse
                            json_clean = re.sub(r"Decimal\('([^']+)'\)", r"\1", json_str)
                            print(f"🔍 Query Agent: JSON sau khi clean: {json_clean[:100]}...")
                            parsed = ast.literal_eval(json_clean)
                            print(f"🔍 Query Agent: Parsed result type: {type(parsed)}, length: {len(parsed) if isinstance(parsed, list) else 'N/A'}")
                            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                                cols = list(parsed[0].keys())
                                rows = [[safe(row.get(c)) for c in cols] for row in parsed]
                                table = {"columns": cols, "data": rows}
                                print(f"✅ Query Agent: Đã parse thành công JSON string thành bảng {len(cols)} cột, {len(rows)} dòng")
                                print(f"✅ Query Agent: Columns: {cols}")
                                print(f"✅ Query Agent: Sample data: {rows[:2] if rows else 'No data'}")
                            else:
                                print(f"⚠️ Query Agent: Parsed data không phải list[dict]: {type(parsed)}")
                        except Exception as e:
                            print(f"⚠️ Query Agent: Không thể parse JSON string: {e}")
                            import traceback
                            print(f"⚠️ Query Agent: Traceback: {traceback.format_exc()}")
                    else:
                        print(f"⚠️ Query Agent: Không thỏa mãn điều kiện parse - columns: {table.get('columns')}, data: {table.get('data')}")
                    
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
                    print(f"🔍 Query Agent: Xử lý object result - type: {type(result)}")
                    result_str = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                    print(f"🔍 Query Agent: Object result string: {result_str[:200]}...")
                    table = to_table(result_str)
                    print(f"🔍 Query Agent: Sau to_table - columns: {table.get('columns')}, data: {table.get('data')}")
                    
                    # CƯỠNG BỨC PARSE: Nếu table chỉ có 1 cột "result" chứa JSON string
                    print(f"🔍 Query Agent (object): Kiểm tra table format - columns: {table.get('columns')}, data length: {len(table.get('data', []))}")
                    if (table.get("columns") == ["result"] and 
                        table.get("data") and 
                        len(table["data"]) == 1 and 
                        isinstance(table["data"][0][0], str) and 
                        table["data"][0][0].strip().startswith('[')):
                        
                        print(f"🔍 Query Agent (object): Phát hiện JSON string trong cột result, bắt đầu parse...")
                        # Thử parse JSON string trong cột result
                        json_str = table["data"][0][0]
                        print(f"🔍 Query Agent (object): JSON string để parse: {json_str[:100]}...")
                        try:
                            import ast
                            # Loại bỏ Decimal('...') để có thể parse
                            json_clean = re.sub(r"Decimal\('([^']+)'\)", r"\1", json_str)
                            print(f"🔍 Query Agent (object): JSON sau khi clean: {json_clean[:100]}...")
                            parsed = ast.literal_eval(json_clean)
                            print(f"🔍 Query Agent (object): Parsed result type: {type(parsed)}, length: {len(parsed) if isinstance(parsed, list) else 'N/A'}")
                            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                                cols = list(parsed[0].keys())
                                rows = [[safe(row.get(c)) for c in cols] for row in parsed]
                                table = {"columns": cols, "data": rows}
                                print(f"✅ Query Agent (object): Đã parse thành công JSON string thành bảng {len(cols)} cột, {len(rows)} dòng")
                                print(f"✅ Query Agent (object): Columns: {cols}")
                                print(f"✅ Query Agent (object): Sample data: {rows[:2] if rows else 'No data'}")
                            else:
                                print(f"⚠️ Query Agent (object): Parsed data không phải list[dict]: {type(parsed)}")
                        except Exception as e:
                            print(f"⚠️ Query Agent (object): Không thể parse JSON string: {e}")
                            import traceback
                            print(f"⚠️ Query Agent (object): Traceback: {traceback.format_exc()}")
                    else:
                        print(f"⚠️ Query Agent (object): Không thỏa mãn điều kiện parse - columns: {table.get('columns')}, data: {table.get('data')}")
                    
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
