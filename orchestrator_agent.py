import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

class OrchestratorAgent:
    """
    Orchestrator Agent - Phân tích intent người dùng bằng LLM và điều phối các agent khác
    """
    
    def __init__(self):
        load_dotenv()
        # KHÔNG khởi tạo LLM ở đây để tránh lỗi "Event loop is closed"
        self.GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.llm_model = "models/gemini-2.5-flash-lite"
        self.llm_temperature = 0.1
        
        # Định nghĩa các agent có sẵn
        self.available_agents = {
            "query_agent": {
                "name": "Query Agent",
                "description": "Truy vấn cơ sở dữ liệu, tìm kiếm thông tin từ database",
                "capabilities": ["database_query", "data_retrieval", "sql_analysis"]
            },
            "cv_agent": {
                "name": "CV Agent", 
                "description": "Phân tích CV, so sánh ứng viên, đánh giá hồ sơ",
                "capabilities": ["cv_analysis", "candidate_matching", "resume_evaluation"]
            },
            "chart_agent": {
                "name": "Chart Agent",
                "description": "Tạo biểu đồ, trực quan hóa dữ liệu, báo cáo thống kê",
                "capabilities": ["data_visualization", "chart_creation", "statistical_analysis"]
            },
            "analysis_agent": {
                "name": "Analysis Agent",
                "description": "Tổng hợp kết quả, phân tích tổng thể, đưa ra insights",
                "capabilities": ["data_synthesis", "insight_generation", "report_creation"]
            }
        }

    async def _prepare_input_for_agent(self, agent_name: str, user_input: str, intent_analysis: Dict[str, Any], accumulated_results: List[Dict[str, Any]] | None = None) -> str:
        """
        Dùng LLM để tạo input phù hợp cho từng agent (không dùng keyword rules).
        """
        # Khởi tạo LLM mỗi lần gọi để tránh lỗi "Event loop is closed"
        llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            google_api_key=self.GEMINI_API_KEY,
            temperature=self.llm_temperature,
        )
        context_prev = "\n".join([
            f"- {r.get('agent')}: {str(r.get('result'))[:500]}" for r in (accumulated_results or [])
        ])

        if agent_name == "query_agent":
            prep_prompt = f"""
Bạn là Orchestrator. Hãy chuyển hoá yêu cầu người dùng thành CHỈ THỊ cho Query Agent.
Yêu cầu người dùng: "{user_input}"
Kế hoạch hiện tại: {intent_analysis}
Ngữ cảnh trước đó (nếu có):\n{context_prev}

Hãy trả về một chuỗi hướng dẫn NGẮN GỌN cho Query Agent với các nguyên tắc:
- Nhiệm vụ duy nhất: truy vấn dữ liệu và trả về bảng với cấu trúc {{"columns": [...], "data": [[...]]}}.
- Không nhắc tới chuyện vẽ biểu đồ, không hỏi thêm thông tin, không hướng dẫn thao tác.
- Nếu yêu cầu cần dữ liệu tổng hợp, hãy mô tả rõ bảng/cột cần lấy theo ngữ nghĩa (ví dụ: nhan_vien, phong_ban, luong_co_ban...).
- Ngôn ngữ: tiếng Việt, súc tích 1-2 câu.
"""
            try:
                resp = await llm.ainvoke(prep_prompt)
                return resp.content if hasattr(resp, 'content') else str(resp)
            except Exception:
                return user_input
        else:
            return user_input
    
    async def analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Phân tích intent của người dùng bằng LLM và xác định agents cần gọi
        """
        try:
            # Khởi tạo LLM mỗi lần gọi để tránh lỗi "Event loop is closed"
            llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                google_api_key=self.GEMINI_API_KEY,
                temperature=self.llm_temperature,
            )
            # Tạo prompt cho LLM phân tích intent
            agents_info = "\n".join([
                f"- {agent_id}: {info['description']} (Capabilities: {', '.join(info['capabilities'])})"
                for agent_id, info in self.available_agents.items()
            ])
            
            prompt = f"""
Bạn là một AI Orchestrator chuyên phân tích yêu cầu của người dùng và quyết định gọi agent nào.

Các agent có sẵn:
{agents_info}

Yêu cầu của người dùng: "{user_input}"

Hãy phân tích và trả về JSON với format:
{{
    "primary_intent": "mô tả intent chính",
    "required_agents": ["agent1", "agent2", ...],
    "execution_plan": [
        {{"step": 1, "agent": "agent_name", "reason": "lý do gọi agent này"}},
        {{"step": 2, "agent": "agent_name", "reason": "lý do gọi agent này"}}
    ],
    "special_requirements": {{
        "needs_data": true/false,
        "needs_chart": true/false,
        "needs_analysis": true/false
    }},
    "confidence": 0.0-1.0,
    "reasoning": "giải thích lý do lựa chọn"
}}

Lưu ý đặc biệt:
- Nếu yêu cầu tạo chart/biểu đồ mà không có dữ liệu, PHẢI gọi query_agent trước để lấy dữ liệu
- Có thể gọi nhiều agent theo thứ tự logic
- Luôn kết thúc bằng analysis_agent để tổng hợp kết quả
"""
            
            # Gọi LLM để phân tích
            response = await llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            try:
                # Tìm JSON trong response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback nếu không tìm thấy JSON
                    result = {
                        "primary_intent": "unknown",
                        "required_agents": ["query_agent"],
                        "execution_plan": [{"step": 1, "agent": "query_agent", "reason": "fallback"}],
                        "special_requirements": {"needs_data": True, "needs_chart": False, "needs_analysis": True},
                        "confidence": 0.5,
                        "reasoning": "LLM response parsing failed, using fallback"
                    }
                
                return result
                
            except json.JSONDecodeError as e:
                print(f" JSON parsing error: {e}")
                # Fallback response
                return {
                    "primary_intent": "unknown",
                    "required_agents": ["query_agent"],
                    "execution_plan": [{"step": 1, "agent": "query_agent", "reason": "fallback due to parsing error"}],
                    "special_requirements": {"needs_data": True, "needs_chart": False, "needs_analysis": True},
                    "confidence": 0.3,
                    "reasoning": f"JSON parsing failed: {e}"
                }
                
        except Exception as e:
            print(f" LLM analysis error: {e}")
            # Fallback response
            return {
                "primary_intent": "error",
                "required_agents": ["query_agent"],
                "execution_plan": [{"step": 1, "agent": "query_agent", "reason": "fallback due to error"}],
                "special_requirements": {"needs_data": True, "needs_chart": False, "needs_analysis": True},
                "confidence": 0.1,
                "reasoning": f"LLM analysis failed: {e}"
            }
    
    async def route_to_agents(self, user_input: str, intent_analysis: Dict[str, Any], uploaded_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Điều phối request đến nhiều agent theo execution plan
        """
        execution_plan = intent_analysis.get("execution_plan", [])
        required_agents = intent_analysis.get("required_agents", [])
        
        routing_result = {
            "orchestrator": {
                "intent_analysis": intent_analysis,
                "execution_plan": execution_plan,
                "timestamp": asyncio.get_event_loop().time()
            },
            "agent_results": [],
            "execution_summary": {
                "total_steps": len(execution_plan),
                "successful_steps": 0,
                "failed_steps": 0
            }
        }
        
        # Lưu trữ kết quả từ các agent để truyền cho agent tiếp theo
        accumulated_data = {"uploaded_files": uploaded_files} if uploaded_files else None
        accumulated_results = []
        
        # Thực thi từng step trong execution plan
        for step_info in execution_plan:
            step_num = step_info.get("step", 0)
            agent_name = step_info.get("agent", "")
            reason = step_info.get("reason", "")
            
            print(f"Step {step_num}: Executing {agent_name} - {reason}")
            
            try:
                # Import và gọi agent tương ứng
                agent_result = await self._call_agent(agent_name, user_input, intent_analysis, accumulated_data, accumulated_results)
                
                # Lưu kết quả
                step_result = {
                    "step": step_num,
                    "agent": agent_name,
                    "reason": reason,
                    "status": agent_result.get("status", "unknown"),
                    "result": agent_result,
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                routing_result["agent_results"].append(step_result)
                accumulated_results.append(agent_result)
                
                # Cập nhật accumulated_data cho agent tiếp theo
                if agent_result.get("status") == "success" and agent_result.get("result"):
                    accumulated_data = agent_result["result"]
                
                routing_result["execution_summary"]["successful_steps"] += 1
                print(f" Step {step_num} completed successfully")
                
            except Exception as e:
                error_result = {
                    "step": step_num,
                    "agent": agent_name,
                    "reason": reason,
                    "status": "error",
                    "error": str(e),
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                routing_result["agent_results"].append(error_result)
                routing_result["execution_summary"]["failed_steps"] += 1
                print(f" Step {step_num} failed: {e}")
        
        return routing_result
    
    async def _call_agent(self, agent_name: str, user_input: str, intent_analysis: Dict[str, Any] | None = None, accumulated_data: Any = None, accumulated_results: List[Dict] = None) -> Dict[str, Any]:
        """
        Gọi agent cụ thể với dữ liệu từ các agent trước đó
        """
        try:
            if agent_name == "query_agent":
                from query_agent import QueryAgent
                agent = QueryAgent()
                prepared = await self._prepare_input_for_agent("query_agent", user_input, intent_analysis or {}, accumulated_results=accumulated_results)
                return await agent.process(prepared)
                
            elif agent_name == "cv_agent":
                from cv_agent import CVAgent
                agent = CVAgent()
                # Truyền uploaded_files nếu có
                uploaded_files = accumulated_data.get("uploaded_files", []) if accumulated_data else []
                return await agent.process(user_input, uploaded_files)
                
            elif agent_name == "chart_agent":
                from chart_agent import ChartAgent
                agent = ChartAgent()
                # Nếu có dữ liệu từ query_agent, truyền vào chart_agent
                if accumulated_data:
                    # Bảo vệ: nếu dữ liệu ở dạng một cột 'result' chứa chuỗi list-dict, cố gắng parse thành bảng
                    try:
                        if isinstance(accumulated_data, dict) and accumulated_data.get("columns") == ["result"]:
                            rows = accumulated_data.get("data") or []
                            if rows and isinstance(rows[0], list) and rows[0]:
                                payload = rows[0][0]
                                import json, ast
                                table = None
                                try:
                                    obj = json.loads(payload)
                                    if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
                                        cols = list(obj[0].keys()) if obj else []
                                        data_rows = [[obj_i.get(c) for c in cols] for obj_i in obj]
                                        table = {"columns": cols, "data": data_rows}
                                except Exception:
                                    try:
                                        lit = ast.literal_eval(payload)
                                        if isinstance(lit, list) and (not lit or isinstance(lit[0], dict)):
                                            cols = list(lit[0].keys()) if lit else []
                                            data_rows = [[lit_i.get(c) for c in cols] for lit_i in lit]
                                            table = {"columns": cols, "data": data_rows}
                                    except Exception:
                                        pass
                                if table:
                                    return await agent.process(user_input, table)
                    except Exception:
                        pass
                    # Truyền dữ liệu thô từ Query Agent
                    if isinstance(accumulated_data, dict) and "raw_result" in accumulated_data:
                        return await agent.process(user_input, accumulated_data["raw_result"])
                    elif isinstance(accumulated_data, dict) and "text" in accumulated_data:
                        return await agent.process(user_input, accumulated_data["text"])
                    else:
                        return await agent.process(user_input, accumulated_data)
                else:
                    return await agent.process(user_input)
                
            elif agent_name == "analysis_agent":
                from analysis_agent import AnalysisAgent
                agent = AnalysisAgent()
                # Truyền tất cả kết quả từ các agent trước đó
                return await agent.process(user_input, accumulated_results or [])
                
            else:
                return {
                    "agent": agent_name,
                    "status": "error",
                    "error": f"Unknown agent: {agent_name}"
                }
                
        except Exception as e:
            return {
                "agent": agent_name,
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def process(self, user_input: str, uploaded_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Xử lý request chính của orchestrator với LLM analysis và multi-agent execution
        """
        print(f" Orchestrator: Phân tích intent cho '{user_input}'")
        
        # Bước 1: Phân tích intent bằng LLM
        intent_analysis = await self.analyze_intent(user_input)
        print(f" Intent Analysis: {intent_analysis}")
        
        # Bước 2: Điều phối đến nhiều agent theo execution plan
        result = await self.route_to_agents(user_input, intent_analysis, uploaded_files)
        
        # Bước 3: Tổng hợp kết quả cuối cùng
        final_result = {
            "orchestrator": result["orchestrator"],
            "execution_summary": result["execution_summary"],
            "agent_results": result["agent_results"],
            "final_status": "success" if result["execution_summary"]["failed_steps"] == 0 else "partial_success",
            "total_agents_executed": len(result["agent_results"]),
            "success_rate": result["execution_summary"]["successful_steps"] / max(result["execution_summary"]["total_steps"], 1)
        }
        
        return final_result

# Test function
async def test_orchestrator():
    orchestrator = OrchestratorAgent()
    
    test_cases = [
        {
            "input": "Tìm nhân viên có lương cao nhất",
            "expected": "Should call query_agent only"
        },
        {
            "input": "Phân tích CV của ứng viên Python developer",
            "expected": "Should call cv_agent only"
        },
        {
            "input": "Tạo biểu đồ thống kê nhân viên theo phòng ban",
            "expected": "Should call query_agent first, then chart_agent"
        },
        {
            "input": "Tổng hợp báo cáo về tình hình nhân sự",
            "expected": "Should call multiple agents and end with analysis_agent"
        },
        {
            "input": "Tạo biểu đồ mà không có dữ liệu",
            "expected": "Should call query_agent first to get data, then chart_agent"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case['input']}")
        print(f"Expected: {test_case['expected']}")
        print(f"{'='*60}")
        
        try:
            result = await orchestrator.process(test_case['input'])
            
            print(f" Execution Summary:")
            print(f"  - Total Steps: {result['execution_summary']['total_steps']}")
            print(f"  - Successful: {result['execution_summary']['successful_steps']}")
            print(f"  - Failed: {result['execution_summary']['failed_steps']}")
            print(f"  - Success Rate: {result['success_rate']:.2%}")
            
            print(f"\n Agent Results:")
            for agent_result in result['agent_results']:
                print(f"  Step {agent_result['step']}: {agent_result['agent']} - {agent_result['status']}")
                if agent_result['status'] == 'error':
                    print(f"    Error: {agent_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f" Test failed: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
