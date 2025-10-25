import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import all agents
from orchestrator_agent import OrchestratorAgent
from query_agent import QueryAgent
from cv_agent import CVAgent
from chart_agent import ChartAgent
from analysis_agent import AnalysisAgent
from conversational_agent import ConversationalAgent

class MultiAgentSystem:
    """
    Hệ thống Multi-Agent chính
    """
    
    def __init__(self):
        load_dotenv()
        
        # Khởi tạo tất cả agents
        self.orchestrator = OrchestratorAgent()
        self.query_agent = QueryAgent()
        self.cv_agent = CVAgent()
        self.chart_agent = ChartAgent()
        self.analysis_agent = AnalysisAgent()
        self.conversational_agent = ConversationalAgent()
        
        # Lịch sử conversation
        self.conversation_history = []
        
        print("Multi-Agent System da khoi tao!")
        print("Cac agent co san:")
        print("  - Orchestrator: Phan tich intent va dieu phoi")
        print("  - Query Agent: Truy van co so du lieu")
        print("  - CV Agent: Phan tich CV va ung vien")
        print("  - Chart Agent: Tao bieu do va truc quan hoa")
        print("  - Analysis Agent: Tong hop va phan tich ket qua")
        print("  - Conversational Agent: Tro chuyen chung va chao hoi")
    
    async def process_single_request(self, user_input: str, uploaded_files: List[str] = None) -> Dict[str, Any]:
        """
        Xử lý một yêu cầu đơn lẻ với orchestrator mới
        """
        try:
            print(f"\n{'='*60}")
            print(f"Xu ly yeu cau: {user_input}")
            if uploaded_files:
                print(f"Uploaded files: {', '.join(uploaded_files)}")
            print(f"{'='*60}")
            
            # Nếu có uploaded files, thêm vào user_input
            if uploaded_files:
                user_input += f" [Uploaded files: {', '.join(uploaded_files)}]"
            
            # Bước 1: Orchestrator phân tích intent và thực thi multi-agent
            print("Orchestrator: Phan tich intent va thuc thi multi-agent...")
            orchestrator_result = await self.orchestrator.process(user_input, uploaded_files)
            
            # Thêm uploaded_files vào result để CV Agent có thể sử dụng
            if uploaded_files:
                orchestrator_result["uploaded_files"] = uploaded_files
            
            # Bước 2: Kiểm tra nếu là conversational response
            if orchestrator_result.get("conversational_response"):
                # Nếu là conversational response, không cần gọi Analysis Agent
                orchestrator_result["analysis_result"] = {
                    "agent": "conversational_agent",
                    "status": "success",
                    "result": {
                        "formatted_summary": orchestrator_result["conversational_response"],
                        "summary_report": {
                            "conversational": True,
                            "response": orchestrator_result["conversational_response"]
                        }
                    }
                }
            else:
                # Bước 2: Luôn gọi Analysis Agent để tạo báo cáo Markdown tổng hợp cho UI
                try:
                    agent_results_for_analysis = orchestrator_result.get("agent_results", [])
                    analysis_result = await self.analysis_agent.process(user_input, agent_results_for_analysis)
                    orchestrator_result["analysis_result"] = analysis_result
                except Exception as _e:
                    # Không chặn luồng nếu phân tích lỗi
                    orchestrator_result["analysis_result"] = {
                        "agent": "analysis_agent",
                        "status": "error",
                        "error": str(_e)
                    }

            # Lưu vào lịch sử
            self.conversation_history.append({
                "user_input": user_input,
                "uploaded_files": uploaded_files,
                "orchestrator_result": orchestrator_result,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return orchestrator_result
            
        except Exception as e:
            return {
                "system": "multi_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def process_with_analysis(self, user_input: str) -> Dict[str, Any]:
        """
        Xử lý yêu cầu với phân tích tổng hợp
        """
        try:
            print(f"\n{'='*60}")
            print(f"Xu ly yeu cau voi phan tich: {user_input}")
            print(f"{'='*60}")
            
            # Bước 1: Orchestrator phân tích intent
            print("Orchestrator: Phan tich intent...")
            orchestrator_result = await self.orchestrator.process(user_input)
            
            # Bước 2: Thu thập kết quả từ các agent
            agent_results = []
            
            # Nếu orchestrator thành công, lấy kết quả từ agent được gọi
            if orchestrator_result.get("agent_result"):
                agent_results.append(orchestrator_result["agent_result"])
            
            # Bước 3: Analysis Agent phân tích tổng hợp
            print(" Analysis Agent: Phân tích tổng hợp...")
            analysis_result = await self.analysis_agent.process(user_input, agent_results)
            
            # Bước 4: Tổng hợp kết quả cuối cùng
            final_result = {
                "system": "multi_agent",
                "status": "success",
                "user_input": user_input,
                "orchestrator_analysis": orchestrator_result,
                "agent_results": agent_results,
                "analysis_result": analysis_result,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Lưu vào lịch sử
            self.conversation_history.append({
                "user_input": user_input,
                "final_result": final_result,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return final_result
            
        except Exception as e:
            return {
                "system": "multi_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def process_workflow(self, user_input: str, workflow_type: str = "auto") -> Dict[str, Any]:
        """
        Xử lý theo workflow cụ thể
        """
        try:
            print(f"\n{'='*60}")
            print(f" Workflow {workflow_type}: {user_input}")
            print(f"{'='*60}")
            
            if workflow_type == "query_then_chart":
                # Workflow: Query -> Chart
                print(" Workflow: Query -> Chart")
                
                # Bước 1: Query Agent
                print(" Query Agent: Truy vấn dữ liệu...")
                query_result = await self.query_agent.process(user_input)
                
                # Bước 2: Chart Agent (nếu query thành công)
                chart_result = None
                if query_result.get("status") == "success" and query_result.get("result"):
                    print(" Chart Agent: Tạo biểu đồ...")
                    chart_result = await self.chart_agent.process("Tạo biểu đồ từ dữ liệu query", query_result.get("result"))
                
                # Bước 3: Analysis Agent
                print(" Analysis Agent: Phân tích tổng hợp...")
                analysis_result = await self.analysis_agent.process(user_input, [query_result, chart_result])
                
                return {
                    "system": "multi_agent",
                    "workflow": "query_then_chart",
                    "status": "success",
                    "query_result": query_result,
                    "chart_result": chart_result,
                    "analysis_result": analysis_result
                }
            
            elif workflow_type == "cv_analysis":
                # Workflow: CV Analysis
                print(" Workflow: CV Analysis")
                
                # Bước 1: CV Agent
                print(" CV Agent: Phân tích CV...")
                cv_result = await self.cv_agent.process(user_input, [])
                
                # Bước 2: Analysis Agent
                print(" Analysis Agent: Phân tích tổng hợp...")
                analysis_result = await self.analysis_agent.process(user_input, [cv_result])
                
                return {
                    "system": "multi_agent",
                    "workflow": "cv_analysis",
                    "status": "success",
                    "cv_result": cv_result,
                    "analysis_result": analysis_result
                }
            
            elif workflow_type == "full_analysis":
                # Workflow: Full Analysis (tất cả agents)
                print(" Workflow: Full Analysis")
                
                # Chạy tất cả agents song song
                print(" Chạy tất cả agents song song...")
                tasks = [
                    self.query_agent.process(user_input),
                    self.cv_agent.process(user_input, []),
                    self.chart_agent.process(user_input)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Lọc kết quả thành công
                successful_results = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
                
                # Analysis Agent
                print(" Analysis Agent: Phân tích tổng hợp...")
                analysis_result = await self.analysis_agent.process(user_input, successful_results)
                
                return {
                    "system": "multi_agent",
                    "workflow": "full_analysis",
                    "status": "success",
                    "all_results": results,
                    "successful_results": successful_results,
                    "analysis_result": analysis_result
                }
            
            else:
                # Mặc định: sử dụng orchestrator
                return await self.process_single_request(user_input)
                
        except Exception as e:
            return {
                "system": "multi_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def show_menu(self):
        """Hiển thị menu"""
        print("\n" + "="*60)
        print(" MULTI-AGENT HR SYSTEM")
        print("="*60)
        print("1. Xử lý yêu cầu thông thường (Orchestrator)")
        print("2. Xử lý với phân tích tổng hợp")
        print("3. Workflow: Query -> Chart")
        print("4. Workflow: CV Analysis")
        print("5. Workflow: Full Analysis (tất cả agents)")
        print("6. Xem lịch sử conversation")
        print("7. Test từng agent riêng lẻ")
        print("0. Thoát")
        print("-"*60)
    
    def show_agent_menu(self):
        """Hiển thị menu test agents"""
        print("\n" + "="*40)
        print("🧪 TEST AGENTS")
        print("="*40)
        print("1. Test Query Agent")
        print("2. Test CV Agent")
        print("3. Test Chart Agent")
        print("4. Test Analysis Agent")
        print("5. Test Orchestrator")
        print("0. Quay lại menu chính")
        print("-"*40)
    
    async def test_individual_agent(self, agent_choice: str, test_input: str):
        """Test agent riêng lẻ"""
        try:
            if agent_choice == "1":  # Query Agent
                print(" Testing Query Agent...")
                result = await self.query_agent.process(test_input)
            elif agent_choice == "2":  # CV Agent
                print(" Testing CV Agent...")
                result = await self.cv_agent.process(test_input)
            elif agent_choice == "3":  # Chart Agent
                print(" Testing Chart Agent...")
                # Tạo dữ liệu test cho chart
                test_data = {
                    "columns": ["PhongBan", "SoLuong"],
                    "data": [["IT", 15], ["HR", 8], ["Finance", 12]]
                }
                result = await self.chart_agent.process(test_input, test_data)
            elif agent_choice == "4":  # Analysis Agent
                print(" Testing Analysis Agent...")
                # Mock results cho test
                mock_results = [
                    {"agent": "query_agent", "status": "success", "result": {"data": "test"}},
                    {"agent": "cv_agent", "status": "success", "result": {"total_cvs": 5}}
                ]
                result = await self.analysis_agent.process(test_input, mock_results)
            elif agent_choice == "5":  # Orchestrator
                print(" Testing Orchestrator...")
                result = await self.orchestrator.process(test_input)
            else:
                result = {"error": "Invalid agent choice"}
            
            print(f"\n Kết quả:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
        except Exception as e:
            print(f" Lỗi test agent: {e}")
    
    async def run_interactive(self):
        """Chạy hệ thống tương tác"""
        print(" Khởi động Multi-Agent HR System...")
        
        while True:
            self.show_menu()
            choice = input("Chọn chức năng (0-7): ").strip()
            
            if choice == "0":
                print(" Tạm biệt!")
                break
            
            elif choice == "1":
                user_input = input("Nhập yêu cầu: ").strip()
                if user_input:
                    result = await self.process_single_request(user_input)
                    print(f"\n Kết quả:")
                    # Convert numpy types to Python types for JSON serialization
                    def convert_numpy_types(obj):
                        if hasattr(obj, 'item'):
                            return obj.item()
                        elif isinstance(obj, dict):
                            return {k: convert_numpy_types(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_numpy_types(item) for item in obj]
                        else:
                            return obj
                    
                    result_clean = convert_numpy_types(result)
                    print(json.dumps(result_clean, ensure_ascii=False, indent=2))
            
            elif choice == "2":
                user_input = input("Nhập yêu cầu: ").strip()
                if user_input:
                    result = await self.process_with_analysis(user_input)
                    print(f"\n Kết quả:")
                    # Convert numpy types to Python types for JSON serialization
                    def convert_numpy_types(obj):
                        if hasattr(obj, 'item'):
                            return obj.item()
                        elif isinstance(obj, dict):
                            return {k: convert_numpy_types(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_numpy_types(item) for item in obj]
                        else:
                            return obj
                    
                    result_clean = convert_numpy_types(result)
                    print(json.dumps(result_clean, ensure_ascii=False, indent=2))
            
            elif choice == "3":
                user_input = input("Nhập yêu cầu query: ").strip()
                if user_input:
                    result = await self.process_workflow(user_input, "query_then_chart")
                    print(f"\n Kết quả:")
                    # Convert numpy types to Python types for JSON serialization
                    def convert_numpy_types(obj):
                        if hasattr(obj, 'item'):
                            return obj.item()
                        elif isinstance(obj, dict):
                            return {k: convert_numpy_types(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_numpy_types(item) for item in obj]
                        else:
                            return obj
                    
                    result_clean = convert_numpy_types(result)
                    print(json.dumps(result_clean, ensure_ascii=False, indent=2))
            
            elif choice == "4":
                user_input = input("Nhập yêu cầu CV analysis: ").strip()
                if user_input:
                    result = await self.process_workflow(user_input, "cv_analysis")
                    print(f"\n Kết quả:")
                    # Convert numpy types to Python types for JSON serialization
                    def convert_numpy_types(obj):
                        if hasattr(obj, 'item'):
                            return obj.item()
                        elif isinstance(obj, dict):
                            return {k: convert_numpy_types(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_numpy_types(item) for item in obj]
                        else:
                            return obj
                    
                    result_clean = convert_numpy_types(result)
                    print(json.dumps(result_clean, ensure_ascii=False, indent=2))
            
            elif choice == "5":
                user_input = input("Nhập yêu cầu full analysis: ").strip()
                if user_input:
                    result = await self.process_workflow(user_input, "full_analysis")
                    print(f"\n Kết quả:")
                    # Convert numpy types to Python types for JSON serialization
                    def convert_numpy_types(obj):
                        if hasattr(obj, 'item'):
                            return obj.item()
                        elif isinstance(obj, dict):
                            return {k: convert_numpy_types(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_numpy_types(item) for item in obj]
                        else:
                            return obj
                    
                    result_clean = convert_numpy_types(result)
                    print(json.dumps(result_clean, ensure_ascii=False, indent=2))
            
            elif choice == "6":
                print(f"\n Lịch sử conversation ({len(self.conversation_history)} entries):")
                for i, entry in enumerate(self.conversation_history[-5:], 1):  # Hiển thị 5 entries gần nhất
                    print(f"{i}. {entry['user_input'][:50]}...")
                    print(f"   Timestamp: {entry['timestamp']}")
            
            elif choice == "7":
                self.show_agent_menu()
                agent_choice = input("Chọn agent để test (0-5): ").strip()
                if agent_choice != "0":
                    test_input = input("Nhập input test: ").strip()
                    if test_input:
                        await self.test_individual_agent(agent_choice, test_input)
            
            else:
                print(" Lựa chọn không hợp lệ!")
            
            print("\n" + "="*60)
    
    async def run_demo(self):
        """Chạy demo với các test cases"""
        print(" Chạy demo Multi-Agent System...")
        
        demo_cases = [
            "Tìm nhân viên có lương cao nhất",
            "Phân tích CV của ứng viên Python developer",
            "Tạo biểu đồ thống kê nhân viên theo phòng ban",
            "So sánh CV với yêu cầu công việc Data Analyst"
        ]
        
        for i, demo_input in enumerate(demo_cases, 1):
            print(f"\n{'='*60}")
            print(f" Demo Case {i}: {demo_input}")
            print(f"{'='*60}")
            
            try:
                result = await self.process_single_request(demo_input)
                print(f" Kết quả:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"Lỗi demo case {i}: {e}")
            
            # Delay giữa các demo
            await asyncio.sleep(2)
        
        print("\n Demo hoàn thành!")

async def main():
    """Main function"""
    system = MultiAgentSystem()
    
    print("\nChọn chế độ chạy:")
    print("1. Interactive Mode")
    print("2. Demo Mode")
    
    mode = input("Chọn chế độ (1-2): ").strip()
    
    if mode == "1":
        await system.run_interactive()
    elif mode == "2":
        await system.run_demo()
    else:
        print(" Chế độ không hợp lệ!")

if __name__ == "__main__":
    asyncio.run(main())
