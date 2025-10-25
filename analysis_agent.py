import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from datetime import datetime

class AnalysisAgent:
    """
    Analysis Agent - Tổng hợp và phân tích kết quả từ các agent khác
    """
    
    def __init__(self):
        load_dotenv()
        self.GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not self.GEMINI_API_KEY:
            raise ValueError("⚠️ Thiếu GOOGLE_API_KEY trong .env")
        # Lưu cấu hình, KHÔNG khởi tạo LLM ở đây để tránh gắn với event loop cũ
        self.llm_model = "models/gemini-2.5-flash-lite"
        self.llm_temperature = 0.3
    
    def _extract_agent_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Trích xuất kết quả từ các agent"""
        agent_results = {
            "query_agent": None,
            "cv_agent": None,
            "chart_agent": None,
            "analysis_agent": None
        }
        
        for result in results:
            agent_name = result.get("agent", "unknown")
            if agent_name in agent_results:
                agent_results[agent_name] = result
        
        return agent_results

    def _list_of_dicts_to_table(self, items: Any) -> Optional[Dict[str, Any]]:
        """Chuyển list[dict] thành bảng {columns, data} và convert Decimal/date/... về JSON-safe.
        Trả về None nếu không phù hợp.
        """
        try:
            from decimal import Decimal  # local import để tránh yêu cầu khi không dùng
        except Exception:
            Decimal = tuple()  # fallback vô hại

        def convert_value(val: Any) -> Any:
            try:
                from datetime import date, datetime
                if isinstance(val, (datetime, date)):
                    return val.isoformat()
            except Exception:
                pass
            if isinstance(val, float):
                return float(val)
            # Decimal
            if isinstance(val, Decimal):
                try:
                    return float(val)
                except Exception:
                    return str(val)
            # set/tuple
            if isinstance(val, (set, tuple)):
                return list(val)
            return val

        if isinstance(items, list) and items and all(isinstance(x, dict) for x in items):
            # Lấy tập cột union để chống thiếu khóa lệch nhau
            columns = []
            seen = set()
            for obj in items:
                for k in obj.keys():
                    if k not in seen:
                        seen.add(k)
                        columns.append(k)
            data = []
            for obj in items:
                row = [convert_value(obj.get(col)) for col in columns]
                data.append(row)
            return {"columns": columns, "data": data}
        return None
    
    def _analyze_data_quality(self, data: Any) -> Dict[str, Any]:
        """Phân tích chất lượng dữ liệu"""
        try:
            if isinstance(data, dict) and "columns" in data and "data" in data:
                df = pd.DataFrame(data["data"], columns=data["columns"])
            elif isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, list) and data:
                df = pd.DataFrame(data)
            else:
                return {"error": "Unsupported data format"}
            
            analysis = {
                "basic_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist()
                },
                "data_quality": {
                    "missing_values": df.isnull().sum().to_dict(),
                    "duplicate_rows": df.duplicated().sum(),
                    "data_types": df.dtypes.to_dict()
                },
                "statistics": {}
            }
            
            # Thống kê cho cột số
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                analysis["statistics"][col] = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "range": df[col].max() - df[col].min()
                }
            
            # Thống kê cho cột phân loại
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                analysis["statistics"][col] = {
                    "unique_values": df[col].nunique(),
                    "most_common": value_counts.head(3).to_dict(),
                    "distribution": value_counts.to_dict()
                }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Data quality analysis failed: {str(e)}"}
    
    def _generate_insights(self, agent_results: Dict[str, Any]) -> List[str]:
        """Tạo insights từ kết quả các agent"""
        insights = []
        
        # Insights từ Query Agent
        query_result = agent_results.get("query_agent")
        if query_result and query_result.get("status") == "success":
            query_data = query_result.get("result", {})
            if isinstance(query_data, dict) and "data" in query_data:
                data_rows = len(query_data.get("data", []))
                if data_rows > 0:
                    insights.append(f"Query Agent trả về {data_rows} bản ghi dữ liệu")
                else:
                    insights.append("Query Agent không tìm thấy dữ liệu phù hợp")
        
        # Insights từ CV Agent
        cv_result = agent_results.get("cv_agent")
        if cv_result and cv_result.get("status") == "success":
            cv_data = cv_result.get("result", {})
            if "total_cvs" in cv_data:
                insights.append(f"CV Agent phân tích {cv_data['total_cvs']} hồ sơ ứng viên")
            if "match_results" in cv_data:
                match_count = len(cv_data["match_results"])
                insights.append(f"CV Agent thực hiện {match_count} so sánh CV-Job")
        
        # Insights từ Chart Agent
        chart_result = agent_results.get("chart_agent")
        if chart_result and chart_result.get("status") == "success":
            chart_data = chart_result.get("result", {})
            if "chart_info" in chart_data:
                chart_file = chart_data["chart_info"].get("chart_file")
                if chart_file:
                    insights.append(f"Chart Agent tạo biểu đồ: {chart_file}")
        
        return insights
    
    def _create_summary_report(self, agent_results: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Tạo báo cáo tổng hợp"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_input,
            "execution_summary": {
                "total_agents": len([r for r in agent_results.values() if r is not None]),
                "successful_agents": len([r for r in agent_results.values() 
                                        if r and r.get("status") == "success"]),
                "failed_agents": len([r for r in agent_results.values() 
                                    if r and r.get("status") == "error"])
            },
            "agent_results": {},
            "insights": [],
            "recommendations": []
        }
        
        # Tổng hợp kết quả từng agent
        for agent_name, result in agent_results.items():
            if result:
                report["agent_results"][agent_name] = {
                    "status": result.get("status"),
                    "summary": self._summarize_agent_result(result)
                }
        
        # Tạo insights
        report["insights"] = self._generate_insights(agent_results)
        
        # Tạo recommendations
        report["recommendations"] = self._generate_recommendations(agent_results)
        
        return report
    
    def _summarize_agent_result(self, result: Dict[str, Any]) -> str:
        """Tóm tắt kết quả của một agent"""
        if result.get("status") == "success":
            agent_name = result.get("agent", "unknown")
            if agent_name == "query_agent":
                return "Truy vấn cơ sở dữ liệu thành công"
            elif agent_name == "cv_agent":
                return "Phân tích CV và ứng viên thành công"
            elif agent_name == "chart_agent":
                return "Tạo biểu đồ thành công"
            else:
                return "Xử lý thành công"
        elif result.get("status") == "error":
            return f"Lỗi: {result.get('error', 'Unknown error')}"
        else:
            return "Trạng thái không xác định"
    
    def _generate_recommendations(self, agent_results: Dict[str, Any]) -> List[str]:
        """Tạo khuyến nghị dựa trên kết quả"""
        recommendations = []
        
        # Kiểm tra lỗi và đưa ra khuyến nghị
        for agent_name, result in agent_results.items():
            if result and result.get("status") == "error":
                error = result.get("error", "")
                if "quota" in error.lower() or "rate limit" in error.lower():
                    recommendations.append("Giảm tần suất gọi API để tránh vượt quota")
                elif "connection" in error.lower():
                    recommendations.append("Kiểm tra kết nối mạng và cấu hình database")
                elif "file not found" in error.lower():
                    recommendations.append("Kiểm tra đường dẫn file và quyền truy cập")
        
        # Khuyến nghị dựa trên kết quả thành công
        query_result = agent_results.get("query_agent")
        if query_result and query_result.get("status") == "success":
            recommendations.append("Có thể tạo biểu đồ để trực quan hóa dữ liệu query")
        
        cv_result = agent_results.get("cv_agent")
        if cv_result and cv_result.get("status") == "success":
            recommendations.append("Có thể so sánh thêm với các tiêu chí khác")
        
        return recommendations
    
    def _summarize_table_for_user(self, table_data: Dict[str, Any]) -> str:
        """Tạo tóm tắt bảng dữ liệu cho người dùng bằng LLM."""
        if not table_data or not table_data.get("data"):
            return "Không có dữ liệu để hiển thị."
        
        columns = table_data.get("columns", [])
        data = table_data.get("data", [])
        
        if not columns or not data:
            return "Dữ liệu không hợp lệ."
        
        # Chuyển đổi dữ liệu thành format dễ đọc cho LLM
        data_summary = []
        for row in data:
            row_dict = {}
            for i, col in enumerate(columns):
                if i < len(row):
                    row_dict[col] = row[i]
            data_summary.append(row_dict)
        
        return f"Dữ liệu gồm {len(data)} dòng với các cột: {', '.join(columns)}. Dữ liệu: {data_summary}"
    
    async def _ai_analysis(self, user_input: str, agent_results: Dict[str, Any], first_table_data: Optional[Dict[str, Any]] = None) -> str:
        """Phân tích bằng AI"""
        try:
            # Khởi tạo LLM MỖI LẦN GỌI để tránh lỗi "Event loop is closed"
            llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                google_api_key=self.GEMINI_API_KEY,
                temperature=self.llm_temperature,
            )
            # Chuẩn bị prompt cho AI
            results_summary = {}
            for agent_name, result in agent_results.items():
                if result:
                    results_summary[agent_name] = {
                        "status": result.get("status"),
                        "has_data": bool(result.get("result")),
                        "error": result.get("error") if result.get("status") == "error" else None
                    }
            
            # Thêm thông tin về dữ liệu bảng nếu có
            table_summary = ""
            if first_table_data and first_table_data.get("data"):
                table_summary = self._summarize_table_for_user(first_table_data)

            prompt = f"""
Bạn là một chuyên gia phân tích dữ liệu HR. Hãy phân tích kết quả từ các agent AI và đưa ra câu trả lời trực tiếp, thân thiện với người dùng.

Yêu cầu người dùng: {user_input}

Kết quả từ các agent:
{json.dumps(results_summary, ensure_ascii=False, indent=2)}

Dữ liệu chính được truy vấn (nếu có):
{table_summary}

Hãy đưa ra:
1. Một câu trả lời trực tiếp, ngắn gọn, thân thiện với người dùng, tóm tắt kết quả chính (ví dụ: "Dựa trên yêu cầu của bạn, đây là thông tin về X: ...").
2. Nếu có dữ liệu bảng, hãy nêu bật các điểm chính từ bảng (ví dụ: "Phòng ban có số lượng nhân viên cao nhất là...", "Mức lương trung bình cao nhất là...").
3. Bất kỳ khuyến nghị hoặc bước tiếp theo nào có thể hữu ích.

Trả lời bằng tiếng Việt, sử dụng Markdown để định dạng đẹp (ví dụ: dùng **bold**, *italic*, list).
"""
            
            response = await llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return f"Không thể thực hiện phân tích AI: {str(e)}"

    async def _final_answer(self, user_input: str, table_markdown: str | None) -> str:
        """Sinh câu trả lời tự nhiên, trực tiếp cho người dùng."""
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                google_api_key=self.GEMINI_API_KEY,
                temperature=0.2,
            )
            context = table_markdown or "(không có bảng dữ liệu hiển thị)"
            prompt = f"""
Bạn là trợ lý nhân sự. Hãy TRẢ LỜI TRỰC TIẾP yêu cầu sau bằng tiếng Việt, ngắn gọn, dễ hiểu, không giải thích kỹ thuật:

Yêu cầu: {user_input}

Dữ liệu liên quan (markdown):
{context}

Yêu cầu định dạng câu trả lời:
- 1-3 câu cô đọng hoặc 3 bullet ngắn; KHÔNG nhắc 'dựa trên dữ liệu', KHÔNG nói về agent, schema, hay kỹ thuật.
- Nếu là so sánh/biểu đồ: nêu 1-2 điểm nổi bật (phòng ban cao nhất/thấp nhất...).
"""
            resp = await llm.ainvoke(prompt)
            return resp.content if hasattr(resp, 'content') else str(resp)
        except Exception:
            return "Không thể sinh câu trả lời tự động lúc này."
    
    async def process(self, user_input: str, agent_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Xử lý phân tích tổng hợp
        """
        try:
            print(f"🔍 Analysis Agent: Phân tích kết quả cho '{user_input}'")
            print(f"🔍 Analysis Agent: Số lượng agent results: {len(agent_results) if agent_results else 0}")
            
            if not agent_results:
                return {
                    "agent": "analysis_agent",
                    "status": "info",
                    "result": {
                        "message": "Analysis Agent sẵn sàng phân tích kết quả",
                        "usage": "Cung cấp kết quả từ các agent khác để phân tích"
                    }
                }
            
            # Debug: In chi tiết từng agent result
            for i, result in enumerate(agent_results):
                print(f"🔍 Analysis Agent: Result {i}: agent={result.get('agent')}, status={result.get('status')}")
                if result.get('agent') == 'query_agent' and result.get('result'):
                    print(f"🔍 Analysis Agent: Query result type: {type(result['result'])}")
                    print(f"🔍 Analysis Agent: Query result content: {str(result['result'])[:200]}...")
            
            # Trích xuất kết quả từ các agent
            extracted_results = self._extract_agent_results(agent_results)
            
            # Tạo báo cáo tổng hợp
            summary_report = self._create_summary_report(extracted_results, user_input)
            
            # Phân tích chất lượng dữ liệu nếu có
            data_quality_analysis = None
            for result in agent_results:
                if result.get("status") == "success" and "result" in result:
                    result_data = result["result"]
                    if isinstance(result_data, dict) and "data" in result_data:
                        data_quality_analysis = self._analyze_data_quality(result_data["data"])
                        break
            
            # Tìm bảng dữ liệu đầu tiên để phân tích
            first_table = None
            for r in agent_results:
                if not r:
                    continue
                res = r.get("result")
                # Trường hợp đã chuẩn bảng
                if isinstance(res, dict) and res.get("columns") and res.get("data"):
                    first_table = res
                    break
                # Trường hợp là list[dict] (ví dụ từ Query Agent trả list-dict với Decimal)
                if isinstance(res, list) and res and all(isinstance(x, dict) for x in res):
                    converted = self._list_of_dicts_to_table(res)
                    if converted and converted.get("data"):
                        first_table = converted
                        break
            
            # Phân tích bằng AI với dữ liệu bảng (nếu có)
            ai_analysis = await self._ai_analysis(user_input, extracted_results, first_table)

            table_md = None
            if first_table:
                cols = first_table.get("columns", [])
                rows = first_table.get("data", [])[:6]
                # Render bảng nhỏ (6 dòng) để LLM tham chiếu
                header = "| " + " | ".join([str(c) for c in cols]) + " |"
                sep = "|" + "---|" * len(cols)
                body = "\n".join(["| " + " | ".join([str(c) for c in r]) + " |" for r in rows])
                table_md = "\n".join([header, sep, body])

            # Sử dụng ai_analysis làm câu trả lời chính
            markdown_summary = f"### ✅ Trả lời\n{ai_analysis}"

            return {
                "agent": "analysis_agent",
                "status": "success",
                "result": {
                    "summary_report": summary_report,
                    "data_quality_analysis": data_quality_analysis,
                    "ai_analysis": ai_analysis,
                    "markdown": markdown_summary,
                    "total_agents_processed": len([r for r in agent_results if r]),
                    "success_rate": len([r for r in agent_results if r and r.get("status") == "success"]) / max(len(agent_results), 1)
                }
            }
            
        except Exception as e:
            return {
                "agent": "analysis_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def compare_results(self, results1: List[Dict[str, Any]], results2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """So sánh kết quả từ hai lần chạy"""
        try:
            comparison = {
                "timestamp": datetime.now().isoformat(),
                "comparison_summary": {
                    "results1_count": len(results1),
                    "results2_count": len(results2),
                    "results1_success": len([r for r in results1 if r and r.get("status") == "success"]),
                    "results2_success": len([r for r in results2 if r and r.get("status") == "success"])
                },
                "differences": [],
                "recommendations": []
            }
            
            # So sánh từng agent
            agents1 = {r.get("agent"): r for r in results1 if r}
            agents2 = {r.get("agent"): r for r in results2 if r}
            
            all_agents = set(agents1.keys()) | set(agents2.keys())
            
            for agent in all_agents:
                result1 = agents1.get(agent)
                result2 = agents2.get(agent)
                
                if result1 and result2:
                    if result1.get("status") != result2.get("status"):
                        comparison["differences"].append(f"{agent}: Status changed from {result1.get('status')} to {result2.get('status')}")
                elif result1 and not result2:
                    comparison["differences"].append(f"{agent}: Present in first run, missing in second")
                elif not result1 and result2:
                    comparison["differences"].append(f"{agent}: Missing in first run, present in second")
            
            return {
                "agent": "analysis_agent",
                "status": "success",
                "result": comparison
            }
            
        except Exception as e:
            return {
                "agent": "analysis_agent",
                "status": "error",
                "error": str(e)
            }

# Test function
async def test_analysis_agent():
    agent = AnalysisAgent()
    
    # Mock results
    mock_results = [
        {
            "agent": "query_agent",
            "status": "success",
            "result": {
                "columns": ["name", "salary"],
                "data": [["John", 5000], ["Jane", 6000]]
            }
        },
        {
            "agent": "cv_agent", 
            "status": "success",
            "result": {
                "total_cvs": 5,
                "match_results": {"job1->cv1": {"match_percentage": 85}}
            }
        }
    ]
    
    test_input = "Phân tích kết quả từ query và CV agent"
    
    print(f"Test: {test_input}")
    result = await agent.process(test_input, mock_results)
    print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_analysis_agent())

