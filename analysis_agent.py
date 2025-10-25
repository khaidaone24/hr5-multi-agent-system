import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Safe imports với fallback
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print(" Warning: langchain_google_genai not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print(" Warning: pandas not available")

from datetime import datetime

class AnalysisAgent:
    """
    Analysis Agent - Tổng hợp và trình bày kết quả từ các agent khác theo format đẹp
    Chỉ tập trung vào việc tổng hợp, phân tích và trình bày kết quả, không thực hiện các tác vụ khác
    """
    
    def __init__(self):
        load_dotenv()
        self.GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        # Không raise exception, chỉ warning
        if not self.GEMINI_API_KEY:
            print(" Warning: GOOGLE_API_KEY not found, AI analysis will be disabled")
            self.GEMINI_API_KEY = None
        
        # Lưu cấu hình, KHÔNG khởi tạo LLM ở đây để tránh gắn với event loop cũ
        self.llm_model = "models/gemini-2.5-flash-lite"
        self.llm_temperature = 0.3
        self.ai_enabled = LANGCHAIN_AVAILABLE and self.GEMINI_API_KEY is not None
    
    def _extract_agent_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Trích xuất và phân loại kết quả từ các agent"""
        print(f"🧠 Analysis Agent: Extracting from {len(results)} results")
        agent_results = {
            "query_agent": None,
            "cv_agent": None,
            "chart_agent": None,
            "analysis_agent": None
        }
        
        for i, result in enumerate(results):
            agent_name = result.get("agent", "unknown")
            print(f"🧠 Analysis Agent: Result {i}: agent={agent_name}, status={result.get('status')}")
            if agent_name in agent_results:
                agent_results[agent_name] = result
                print(f"🧠 Analysis Agent: Added {agent_name} to results")
        
        print(f"🧠 Analysis Agent: Final extracted results: {list(agent_results.keys())}")
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
            # Kiểm tra pandas có available không
            if not PANDAS_AVAILABLE:
                return {"error": "Pandas not available for data analysis"}
            
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
        """Tạo báo cáo tổng hợp với format đẹp mắt"""
        # Đếm số lượng agent thành công
        successful_agents = [r for r in agent_results.values() if r and r.get("status") == "success"]
        failed_agents = [r for r in agent_results.values() if r and r.get("status") == "error"]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_input,
            "execution_summary": {
                "total_agents": len([r for r in agent_results.values() if r is not None]),
                "successful_agents": len(successful_agents),
                "failed_agents": len(failed_agents),
                "success_rate": f"{len(successful_agents)}/{len([r for r in agent_results.values() if r is not None])}"
            },
            "agent_results": {},
            "key_findings": [],
            "formatted_summary": ""
        }
        
        # Tổng hợp kết quả từng agent với format đẹp
        for agent_name, result in agent_results.items():
            if result:
                report["agent_results"][agent_name] = {
                    "status": result.get("status"),
                    "summary": self._summarize_agent_result(result),
                    "key_data": self._extract_key_data(result)
                }
        
        # Tạo key findings từ kết quả
        report["key_findings"] = self._generate_key_findings(agent_results)
        
        # Tạo formatted summary
        report["formatted_summary"] = self._create_formatted_summary(agent_results, user_input)
        
        return report
    
    def _extract_key_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Trích xuất dữ liệu quan trọng từ kết quả agent"""
        key_data = {
            "data_type": "unknown",
            "data_summary": "",
            "metrics": {},
            "files_created": []
        }
        
        if result.get("status") != "success":
            return key_data
        
        agent_name = result.get("agent", "unknown")
        result_data = result.get("result", {})
        
        if agent_name == "query_agent":
            if isinstance(result_data, dict) and "data" in result_data:
                data_rows = result_data.get("data", [])
                key_data["data_type"] = "database_query"
                key_data["data_summary"] = f"Truy vấn trả về {len(data_rows)} bản ghi"
                key_data["metrics"]["record_count"] = len(data_rows)
            elif result.get("final_answer"):
                key_data["data_type"] = "text_response"
                key_data["data_summary"] = "Câu trả lời từ cơ sở dữ liệu"
        
        elif agent_name == "cv_agent":
            if "cv_evaluations" in result_data:
                cv_count = len(result_data.get("cv_evaluations", []))
                key_data["data_type"] = "cv_analysis"
                key_data["data_summary"] = f"Phân tích {cv_count} CV"
                key_data["metrics"]["cv_count"] = cv_count
                
                # Lưu toàn bộ CV evaluations để hiển thị đầy đủ
                key_data["full_cv_data"] = result_data.get("cv_evaluations", [])
                
                # Tìm CV có điểm cao nhất
                best_scores = []
                for evaluation in result_data.get("cv_evaluations", []):
                    if evaluation.get("best_match"):
                        score = evaluation["best_match"].get("score", 0)
                        best_scores.append(score)
                
                if best_scores:
                    key_data["metrics"]["highest_score"] = max(best_scores)
                    key_data["metrics"]["average_score"] = sum(best_scores) / len(best_scores)
        
        elif agent_name == "chart_agent":
            if "chart_info" in result_data:
                chart_info = result_data["chart_info"]
                key_data["data_type"] = "chart_visualization"
                key_data["data_summary"] = f"Tạo biểu đồ {chart_info.get('chart_type', 'unknown')}"
                if chart_info.get("chart_file"):
                    key_data["files_created"].append(chart_info["chart_file"])
        
        return key_data
    
    def _summarize_agent_result(self, result: Dict[str, Any]) -> str:
        """Tóm tắt kết quả của một agent với thông tin chi tiết"""
        if result.get("status") == "success":
            agent_name = result.get("agent", "unknown")
            key_data = self._extract_key_data(result)
            
            if agent_name == "query_agent":
                if key_data["metrics"].get("record_count"):
                    return f"✅ Truy vấn thành công: {key_data['data_summary']}"
                else:
                    return "✅ Truy vấn cơ sở dữ liệu thành công"
            elif agent_name == "cv_agent":
                if key_data["metrics"].get("cv_count"):
                    return f"✅ Phân tích CV thành công: {key_data['data_summary']}"
                else:
                    return "✅ Phân tích CV và ứng viên thành công"
            elif agent_name == "chart_agent":
                if key_data["files_created"]:
                    return f"✅ Tạo biểu đồ thành công: {key_data['data_summary']}"
                else:
                    return "✅ Tạo biểu đồ thành công"
            else:
                return "✅ Xử lý thành công"
        elif result.get("status") == "error":
            return f"❌ Lỗi: {result.get('error', 'Unknown error')}"
        else:
            return "⚠️ Trạng thái không xác định"
    
    def _generate_key_findings(self, agent_results: Dict[str, Any]) -> List[str]:
        """Tạo key findings từ kết quả các agent"""
        findings = []
        
        # Findings từ Query Agent
        query_result = agent_results.get("query_agent")
        if query_result and query_result.get("status") == "success":
            key_data = self._extract_key_data(query_result)
            if key_data["metrics"].get("record_count"):
                findings.append(f"📊 Truy vấn dữ liệu: Tìm thấy {key_data['metrics']['record_count']} bản ghi")
            elif query_result.get("final_answer"):
                findings.append("📊 Truy vấn dữ liệu: Có câu trả lời từ cơ sở dữ liệu")
        
        # Findings từ CV Agent
        cv_result = agent_results.get("cv_agent")
        if cv_result and cv_result.get("status") == "success":
            key_data = self._extract_key_data(cv_result)
            if key_data["metrics"].get("cv_count"):
                findings.append(f"👥 Phân tích CV: Đã đánh giá {key_data['metrics']['cv_count']} hồ sơ")
                if key_data["metrics"].get("highest_score"):
                    findings.append(f"⭐ Điểm cao nhất: {key_data['metrics']['highest_score']}%")
                if key_data["metrics"].get("average_score"):
                    findings.append(f"📈 Điểm trung bình: {key_data['metrics']['average_score']:.1f}%")
        
        # Findings từ Chart Agent
        chart_result = agent_results.get("chart_agent")
        if chart_result and chart_result.get("status") == "success":
            key_data = self._extract_key_data(chart_result)
            if key_data["files_created"]:
                findings.append(f"📈 Trực quan hóa: Đã tạo {len(key_data['files_created'])} biểu đồ")
        
        return findings
    
    def _create_formatted_summary(self, agent_results: Dict[str, Any], user_input: str) -> str:
        """Tạo summary được format đẹp mắt"""
        print(f"🧠 Analysis Agent: Creating formatted summary for {len(agent_results)} agents")
        summary_parts = []
        
        # Header
        summary_parts.append("## 📋 Báo Cáo Tổng Hợp")
        summary_parts.append(f"**Yêu cầu:** {user_input}")
        summary_parts.append(f"**Thời gian:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        summary_parts.append("")
        
        # Execution Summary
        successful_count = len([r for r in agent_results.values() if r and r.get("status") == "success"])
        total_count = len([r for r in agent_results.values() if r is not None])
        
        summary_parts.append("### 🎯 Tóm Tắt Thực Hiện")
        summary_parts.append(f"- **Tổng số agent:** {total_count}")
        summary_parts.append(f"- **Thành công:** {successful_count}")
        summary_parts.append(f"- **Tỷ lệ thành công:** {(successful_count/total_count*100):.1f}%" if total_count > 0 else "- **Tỷ lệ thành công:** 0%")
        summary_parts.append("")
        
        # Key Findings
        key_findings = self._generate_key_findings(agent_results)
        if key_findings:
            summary_parts.append("### 🔍 Phát Hiện Chính")
            for finding in key_findings:
                summary_parts.append(f"- {finding}")
            summary_parts.append("")
        
        # Agent Results
        summary_parts.append("### 📊 Kết Quả Chi Tiết")
        for agent_name, result in agent_results.items():
            if result:
                status_icon = "✅" if result.get("status") == "success" else "❌" if result.get("status") == "error" else "⚠️"
                agent_display_name = {
                    "query_agent": "🔍 Query Agent",
                    "cv_agent": "👥 CV Agent", 
                    "chart_agent": "📈 Chart Agent",
                    "analysis_agent": "🧠 Analysis Agent"
                }.get(agent_name, f"🤖 {agent_name}")
                
                summary_parts.append(f"#### {status_icon} {agent_display_name}")
                summary_parts.append(f"- **Trạng thái:** {self._summarize_agent_result(result)}")
                
                # Thêm thông tin chi tiết nếu có
                key_data = self._extract_key_data(result)
                if key_data["metrics"]:
                    for metric, value in key_data["metrics"].items():
                        summary_parts.append(f"- **{metric.replace('_', ' ').title()}:** {value}")
                
                if key_data["files_created"]:
                    summary_parts.append(f"- **Files tạo:** {', '.join(key_data['files_created'])}")
                
                # Hiển thị full CV data với format đẹp như dashboard
                if agent_name == "cv_agent" and key_data.get("full_cv_data"):
                    summary_parts.append("")
                    summary_parts.append("##### 📋 Báo Cáo Đánh Giá Ứng Viên")
                    
                    for i, evaluation in enumerate(key_data["full_cv_data"], 1):
                        cv_name = evaluation.get("cv_name", f"CV_{i}")
                        status = evaluation.get("status", "Unknown")
                        cv_key_info = evaluation.get("cv_key_info", {})
                        
                        # Thông tin ứng viên
                        summary_parts.append(f"**👤 Thông tin ứng viên: {cv_name}**")
                        summary_parts.append(f"- **Trạng thái:** {status}")
                        
                        # Kiểm tra lỗi 429
                        if status == "error" and evaluation.get("error"):
                            error_msg = evaluation.get("error", "")
                            if "429" in error_msg or "Rate limit" in error_msg:
                                summary_parts.append("")
                                summary_parts.append("🚨🚨🚨 **LỖI RATE LIMIT 429** 🚨🚨🚨")
                                summary_parts.append(f"❌ **Lỗi:** {error_msg}")
                                summary_parts.append("⏰ **Thời gian:** " + datetime.now().strftime('%H:%M:%S'))
                                summary_parts.append("🛑 **Hệ thống đã dừng phân tích để tránh lỗi API**")
                                summary_parts.append("💡 **Giải pháp:** Vui lòng thử lại sau 1-2 phút")
                                summary_parts.append("🤖 **Model:** Gemini 2.5 Flash Lite")
                                summary_parts.append("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
                                summary_parts.append("")
                                return "\n".join(summary_parts)
                        
                        if cv_key_info.get("experience_years"):
                            summary_parts.append(f"- **Kinh nghiệm:** {cv_key_info.get('experience_years')} năm")
                        if cv_key_info.get("skills"):
                            summary_parts.append(f"- **Kỹ năng:** {', '.join(cv_key_info.get('skills', [])[:5])}")
                        summary_parts.append("")
                        
                        if evaluation.get("best_match"):
                            best_match = evaluation["best_match"]
                            job_title = best_match.get("job_title", "Unknown")
                            score = best_match.get("score", 0)
                            analysis = best_match.get("analysis", "")
                            
                            # Điểm tổng thể với màu sắc
                            score_color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
                            summary_parts.append(f"**🎯 Xếp hạng và đánh giá bởi AI (beta)**")
                            summary_parts.append(f"**Điểm tổng thể: {score_color} {score}%**")
                            summary_parts.append("")
                            
                            # Phân tích chi tiết từng tiêu chí
                            if best_match.get("detailed_scores"):
                                detailed_scores = best_match["detailed_scores"]
                                summary_parts.append("**📊 Phân tích chi tiết:**")
                                
                                # Job Title
                                if "job_title" in detailed_scores:
                                    job_score = detailed_scores["job_title"].get("score", 0)
                                    job_analysis = detailed_scores["job_title"].get("analysis", "")
                                    summary_parts.append(f"- **Chức danh ({job_score}%):** {job_analysis}")
                                
                                # Skills
                                if "skills" in detailed_scores:
                                    skills_score = detailed_scores["skills"].get("score", 0)
                                    skills_analysis = detailed_scores["skills"].get("analysis", "")
                                    summary_parts.append(f"- **Kỹ năng ({skills_score}%):** {skills_analysis}")
                                
                                # Experience
                                if "experience" in detailed_scores:
                                    exp_score = detailed_scores["experience"].get("score", 0)
                                    exp_analysis = detailed_scores["experience"].get("analysis", "")
                                    summary_parts.append(f"- **Kinh nghiệm ({exp_score}%):** {exp_analysis}")
                                
                                # Education
                                if "education" in detailed_scores:
                                    edu_score = detailed_scores["education"].get("score", 0)
                                    edu_analysis = detailed_scores["education"].get("analysis", "")
                                    summary_parts.append(f"- **Học vấn ({edu_score}%):** {edu_analysis}")
                                
                                summary_parts.append("")
                            
                            # Điểm mạnh và điểm yếu
                            if best_match.get("strengths"):
                                summary_parts.append("**✅ Điểm mạnh:**")
                                for strength in best_match["strengths"]:
                                    summary_parts.append(f"- {strength}")
                                summary_parts.append("")
                            
                            if best_match.get("weaknesses"):
                                summary_parts.append("**❌ Điểm cần cải thiện:**")
                                for weakness in best_match["weaknesses"]:
                                    summary_parts.append(f"- {weakness}")
                                summary_parts.append("")
                            
                            # Tạo biểu đồ donut chart thật
                            summary_parts.append("**📈 Biểu đồ đánh giá:**")
                            suitable_percent = score
                            unsuitable_percent = 100 - score
                            
                            # Tạo donut chart thật bằng Chart Agent
                            try:
                                from chart_agent import ChartAgent
                                chart_agent = ChartAgent()
                                donut_result = chart_agent._create_donut_chart(
                                    suitable_percent, 
                                    unsuitable_percent, 
                                    f"Đánh Giá CV: {cv_name}"
                                )
                                
                                if "chart_file" in donut_result:
                                    chart_file = donut_result["chart_file"]
                                    summary_parts.append(f"![Donut Chart]({chart_file})")
                                    summary_parts.append(f"*Biểu đồ: {donut_result.get('title', 'Đánh giá phù hợp')}*")
                                else:
                                    # Fallback to text chart
                                    summary_parts.append("```")
                                    summary_parts.append(f"🔴 Phù hợp: {suitable_percent}%")
                                    summary_parts.append(f"🟢 Không phù hợp: {unsuitable_percent}%")
                                    summary_parts.append("```")
                            except Exception as e:
                                # Fallback to text chart nếu có lỗi
                                summary_parts.append("```")
                                summary_parts.append(f"🔴 Phù hợp: {suitable_percent}%")
                                summary_parts.append(f"🟢 Không phù hợp: {unsuitable_percent}%")
                                summary_parts.append("```")
                            
                            summary_parts.append("")
                        
                        # Tất cả đánh giá chi tiết
                        if evaluation.get("all_evaluations"):
                            summary_parts.append("**📋 CHI TIẾT ĐÁNH GIÁ TẤT CẢ VỊ TRÍ:**")
                            summary_parts.append("")
                            
                            for eval_item in evaluation["all_evaluations"]:
                                eval_job = eval_item.get("job_title", "Unknown")
                                eval_score = eval_item.get("score", 0)
                                eval_analysis = eval_item.get("analysis", "")
                                eval_color = "🟢" if eval_score >= 70 else "🟡" if eval_score >= 50 else "🔴"
                                
                                summary_parts.append(f"**🎯 {eval_job}**")
                                summary_parts.append(f"- **Điểm số:** {eval_color} {eval_score}%")
                                summary_parts.append(f"- **Phân tích:** {eval_analysis}")
                                
                                # Hiển thị detailed scores nếu có
                                if eval_item.get("detailed_scores"):
                                    summary_parts.append("- **Phân tích chi tiết:**")
                                    for criteria, data in eval_item["detailed_scores"].items():
                                        criteria_name = {
                                            "job_title": "Chức danh",
                                            "skills": "Kỹ năng", 
                                            "experience": "Kinh nghiệm",
                                            "education": "Học vấn"
                                        }.get(criteria, criteria)
                                        criteria_score = data.get("score", 0)
                                        criteria_analysis = data.get("analysis", "")
                                        summary_parts.append(f"  - {criteria_name} ({criteria_score}%): {criteria_analysis}")
                                
                                # Hiển thị strengths và weaknesses
                                if eval_item.get("strengths"):
                                    summary_parts.append("- **Điểm mạnh:**")
                                    for strength in eval_item["strengths"]:
                                        summary_parts.append(f"  + {strength}")
                                
                                if eval_item.get("weaknesses"):
                                    summary_parts.append("- **Điểm cần cải thiện:**")
                                    for weakness in eval_item["weaknesses"]:
                                        summary_parts.append(f"  - {weakness}")
                                
                                summary_parts.append("")
                                summary_parts.append("---")
                                summary_parts.append("")
                        
                        summary_parts.append("---")
                        summary_parts.append("")
                
                summary_parts.append("")
        
        result = "\n".join(summary_parts)
        print(f"🧠 Analysis Agent: Formatted summary created, length: {len(result)}")
        print(f"🧠 Analysis Agent: Summary preview: {result[:300]}...")
        return result
    
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
            # Kiểm tra AI có enabled không
            if not self.ai_enabled:
                return "AI analysis is disabled (missing API key or dependencies)"
            
            # Khởi tạo LLM MỖI LẦN GỌI để tránh lỗi "Event loop is closed"
            llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                google_api_key=self.GEMINI_API_KEY,
                temperature=self.llm_temperature,
            )
            # Chuẩn bị prompt cho AI
            results_summary = {}
            query_agent_answer = None
            cv_agent_answer = None
            
            for agent_name, result in agent_results.items():
                if result:
                    results_summary[agent_name] = {
                        "status": result.get("status"),
                        "has_data": bool(result.get("result")),
                        "error": result.get("error") if result.get("status") == "error" else None
                    }
                    # Ưu tiên sử dụng final_answer từ QueryAgent
                    if agent_name == "query_agent" and result.get("final_answer"):
                        query_agent_answer = result.get("final_answer")
                    # Xử lý CV Agent results - hiển thị đầy đủ
                    elif agent_name == "cv_agent" and result.get("result"):
                        cv_result = result["result"]
                        print(f"    CV Agent result type: {type(cv_result)}")
                        print(f"    CV Agent result keys: {list(cv_result.keys()) if isinstance(cv_result, dict) else 'Not dict'}")
                        
                        # Kiểm tra nếu có cấu trúc nested (từ orchestrator)
                        if isinstance(cv_result, dict) and "result" in cv_result:
                            print(f"    CV Agent có nested result structure")
                            nested_result = cv_result.get("result", {})
                            print(f"    Nested result keys: {list(nested_result.keys()) if isinstance(nested_result, dict) else 'Not dict'}")
                            
                            if isinstance(nested_result, dict) and "cv_evaluations" in nested_result:
                                cv_evaluations = nested_result.get("cv_evaluations", [])
                                print(f"    CV evaluations count (nested): {len(cv_evaluations)}")
                                
                                cv_summary = []
                                cv_summary.append(f"📋 **KẾT QUẢ PHÂN TÍCH CV CHI TIẾT**")
                                cv_summary.append(f"Tổng số CV đã phân tích: {len(cv_evaluations)}")
                                cv_summary.append("")
                                
                                for i, evaluation in enumerate(cv_evaluations, 1):
                                    cv_name = evaluation.get("cv_name", f"CV_{i}")
                                    status = evaluation.get("status", "Unknown")
                                    
                                    print(f"      CV {i}: {cv_name} - {status}")
                                    
                                    cv_summary.append(f"**{i}. {cv_name}**")
                                    cv_summary.append(f"Trạng thái: {status}")
                                    
                                    if evaluation.get("best_match"):
                                        best_match = evaluation["best_match"]
                                        job_title = best_match.get("job_title", "Unknown")
                                        score = best_match.get("score", 0)
                                        analysis = best_match.get("analysis", "")
                                        
                                        print(f"        Best match: {job_title} ({score}%)")
                                        
                                        cv_summary.append(f"🎯 **Phù hợp nhất với:** {job_title}")
                                        cv_summary.append(f"⭐ **Điểm số:** {score}%")
                                        cv_summary.append(f"📝 **Phân tích chi tiết:** {analysis}")
                                        
                                        # Hiển thị tất cả đánh giá nếu có
                                        if evaluation.get("all_evaluations"):
                                            cv_summary.append("📊 **Tất cả đánh giá:**")
                                            for eval_item in evaluation["all_evaluations"]:
                                                eval_job = eval_item.get("job_title", "Unknown")
                                                eval_score = eval_item.get("score", 0)
                                                cv_summary.append(f"  - {eval_job}: {eval_score}%")
                                    
                                    cv_summary.append("")
                                
                                cv_agent_answer = "\n".join(cv_summary)
                                print(f"    CV Agent answer length (nested): {len(cv_agent_answer)}")
                            else:
                                print(f"    Nested result không có cv_evaluations")
                                cv_agent_answer = "CV Agent đã xử lý nhưng chưa có kết quả chi tiết (nested)"
                        elif isinstance(cv_result, dict) and "cv_evaluations" in cv_result:
                            cv_evaluations = cv_result.get("cv_evaluations", [])
                            print(f"    CV evaluations count: {len(cv_evaluations)}")
                            
                            cv_summary = []
                            cv_summary.append(f"📋 **KẾT QUẢ PHÂN TÍCH CV CHI TIẾT**")
                            cv_summary.append(f"Tổng số CV đã phân tích: {len(cv_evaluations)}")
                            cv_summary.append("")
                            
                            for i, evaluation in enumerate(cv_evaluations, 1):
                                cv_name = evaluation.get("cv_name", f"CV_{i}")
                                status = evaluation.get("status", "Unknown")
                                
                                print(f"      CV {i}: {cv_name} - {status}")
                                
                                cv_summary.append(f"**{i}. {cv_name}**")
                                cv_summary.append(f"Trạng thái: {status}")
                                
                                if evaluation.get("best_match"):
                                    best_match = evaluation["best_match"]
                                    job_title = best_match.get("job_title", "Unknown")
                                    score = best_match.get("score", 0)
                                    analysis = best_match.get("analysis", "")
                                    
                                    print(f"        Best match: {job_title} ({score}%)")
                                    
                                    cv_summary.append(f"🎯 **Phù hợp nhất với:** {job_title}")
                                    cv_summary.append(f"⭐ **Điểm số:** {score}%")
                                    cv_summary.append(f"📝 **Phân tích chi tiết:** {analysis}")
                                    
                                    # Hiển thị tất cả đánh giá chi tiết
                                    if evaluation.get("all_evaluations"):
                                        cv_summary.append("📊 **CHI TIẾT ĐÁNH GIÁ TẤT CẢ VỊ TRÍ:**")
                                        cv_summary.append("")
                                        
                                        for eval_item in evaluation["all_evaluations"]:
                                            eval_job = eval_item.get("job_title", "Unknown")
                                            eval_score = eval_item.get("score", 0)
                                            eval_analysis = eval_item.get("analysis", "")
                                            eval_color = "🟢" if eval_score >= 70 else "🟡" if eval_score >= 50 else "🔴"
                                            
                                            cv_summary.append(f"**🎯 {eval_job}**")
                                            cv_summary.append(f"- **Điểm số:** {eval_color} {eval_score}%")
                                            cv_summary.append(f"- **Phân tích:** {eval_analysis}")
                                            
                                            # Hiển thị detailed scores
                                            if eval_item.get("detailed_scores"):
                                                cv_summary.append("- **Phân tích chi tiết:**")
                                                for criteria, data in eval_item["detailed_scores"].items():
                                                    criteria_name = {
                                                        "job_title": "Chức danh",
                                                        "skills": "Kỹ năng", 
                                                        "experience": "Kinh nghiệm",
                                                        "education": "Học vấn"
                                                    }.get(criteria, criteria)
                                                    criteria_score = data.get("score", 0)
                                                    criteria_analysis = data.get("analysis", "")
                                                    cv_summary.append(f"  - {criteria_name} ({criteria_score}%): {criteria_analysis}")
                                            
                                            # Hiển thị strengths và weaknesses
                                            if eval_item.get("strengths"):
                                                cv_summary.append("- **Điểm mạnh:**")
                                                for strength in eval_item["strengths"]:
                                                    cv_summary.append(f"  + {strength}")
                                            
                                            if eval_item.get("weaknesses"):
                                                cv_summary.append("- **Điểm cần cải thiện:**")
                                                for weakness in eval_item["weaknesses"]:
                                                    cv_summary.append(f"  - {weakness}")
                                            
                                            cv_summary.append("")
                                            cv_summary.append("---")
                                            cv_summary.append("")
                                
                                cv_summary.append("")
                            
                            # Thêm biểu đồ donut chart cho best match
                            if evaluation.get("best_match"):
                                best_match = evaluation["best_match"]
                                score = best_match.get("score", 0)
                                suitable_percent = score
                                unsuitable_percent = 100 - score
                                
                                try:
                                    from chart_agent import ChartAgent
                                    chart_agent = ChartAgent()
                                    donut_result = chart_agent._create_donut_chart(
                                        suitable_percent, 
                                        unsuitable_percent, 
                                        f"Đánh Giá CV: {cv_name}"
                                    )
                                    
                                    if "chart_file" in donut_result:
                                        chart_file = donut_result["chart_file"]
                                        cv_summary.append("**📈 Biểu đồ đánh giá:**")
                                        cv_summary.append(f"![Donut Chart]({chart_file})")
                                        cv_summary.append(f"*Biểu đồ: {donut_result.get('title', 'Đánh giá phù hợp')}*")
                                        cv_summary.append("")
                                    else:
                                        # Fallback to text chart
                                        cv_summary.append("**📈 Biểu đồ đánh giá:**")
                                        cv_summary.append("```")
                                        cv_summary.append(f"🔴 Phù hợp: {suitable_percent}%")
                                        cv_summary.append(f"🟢 Không phù hợp: {unsuitable_percent}%")
                                        cv_summary.append("```")
                                        cv_summary.append("")
                                except Exception as e:
                                    print(f"    Lỗi tạo biểu đồ: {e}")
                                    # Fallback to text chart
                                    cv_summary.append("**📈 Biểu đồ đánh giá:**")
                                    cv_summary.append("```")
                                    cv_summary.append(f"🔴 Phù hợp: {suitable_percent}%")
                                    cv_summary.append(f"🟢 Không phù hợp: {unsuitable_percent}%")
                                    cv_summary.append("```")
                                    cv_summary.append("")
                            
                            cv_agent_answer = "\n".join(cv_summary)
                            print(f"    CV Agent answer length: {len(cv_agent_answer)}")
                        else:
                            print(f"    CV Agent result không có cv_evaluations hoặc không phải dict")
                            print(f"    CV Agent result content: {cv_result}")
                            
                            # Thử tìm dữ liệu CV trong cấu trúc khác
                            if isinstance(cv_result, dict):
                                # Kiểm tra các key có thể chứa dữ liệu CV
                                possible_keys = ['cv_evaluations', 'evaluations', 'cv_data', 'results']
                                found_data = False
                                
                                for key in possible_keys:
                                    if key in cv_result and cv_result[key]:
                                        print(f"    Tìm thấy dữ liệu CV trong key: {key}")
                                        found_data = True
                                        break
                                
                                if not found_data:
                                    cv_agent_answer = f"CV Agent đã xử lý nhưng cấu trúc dữ liệu không mong đợi. Keys có sẵn: {list(cv_result.keys())}"
                                else:
                                    cv_agent_answer = "CV Agent đã xử lý nhưng chưa có kết quả chi tiết"
                            else:
                                cv_agent_answer = f"CV Agent đã xử lý nhưng kết quả không phải dict. Type: {type(cv_result)}"
            
            # Thêm thông tin về dữ liệu bảng nếu có
            table_summary = ""
            if first_table_data and first_table_data.get("data"):
                table_summary = self._summarize_table_for_user(first_table_data)

            prompt = """
Bạn là một chuyên gia phân tích dữ liệu HR. Hãy trả lời câu hỏi của người dùng một cách tự nhiên và hữu ích.

Yêu cầu người dùng: {user_input}

Kết quả từ QueryAgent (nếu có):
{query_agent_answer}

Kết quả từ CV Agent (nếu có):
{cv_agent_answer}

Kết quả từ các agent khác:
{results_summary}

Dữ liệu chính được truy vấn (nếu có):
{table_summary}

HƯỚNG DẪN TRẢ LỜI:
1. ƯU TIÊN sử dụng kết quả từ QueryAgent nếu có
2. Nếu có CV Agent results, HIỂN THỊ ĐẦY ĐỦ tất cả thông tin CV (KHÔNG tóm tắt)
3. VỚI CV RESULTS: Hiển thị CHI TIẾT từng vị trí với điểm số, phân tích, strengths, weaknesses
4. BẮT BUỘC: Hiển thị tất cả detailed_scores, strengths, weaknesses cho từng job position
5. KHÔNG được tóm tắt - phải hiển thị đầy đủ thông tin từ CV Agent
6. Sử dụng dữ liệu cụ thể từ kết quả
7. Trả lời tự nhiên như đang nói chuyện
8. Nếu có dữ liệu bảng, nêu các điểm chính
9. VỚI CV: Hiển thị từng job position với đầy đủ thông tin chi tiết

QUAN TRỌNG:
- CHỈ sử dụng dữ liệu thật từ kết quả agent, KHÔNG tạo dữ liệu giả lập
- Nếu không có dữ liệu cụ thể, hãy nói rõ "Chưa có dữ liệu cụ thể"
- KHÔNG được bịa đặt thông tin cá nhân như tên, email, số điện thoại
- CHỈ hiển thị thông tin có trong kết quả agent
- VỚI CV: BẮT BUỘC hiển thị đầy đủ detailed_scores, strengths, weaknesses cho TẤT CẢ job positions
- KHÔNG được tóm tắt CV results - phải hiển thị chi tiết từng vị trí

VÍ DỤ:
- Người dùng hỏi: "Có bao nhiêu nhân viên?"
- QueryAgent trả về: "Công ty hiện có 25 nhân viên"
- Trả lời: "Công ty hiện có **25 nhân viên**. Đây là tổng số nhân viên đang làm việc tại công ty."

- Người dùng hỏi: "Quét CV này"
- CV Agent trả về: "CV_John.pdf phù hợp nhất với Business Analyst (85%)"
- Trả lời: "Đã phân tích CV của bạn. **Kết quả đánh giá chi tiết**:

**🎯 Business Analyst**
- **Điểm số:** 🟢 85%
- **Phân tích:** [Phân tích chi tiết từ CV Agent]
- **Phân tích chi tiết:**
  - Chức danh (80%): [Phân tích chức danh từ CV Agent]
  - Kỹ năng (90%): [Phân tích kỹ năng từ CV Agent]
  - Kinh nghiệm (75%): [Phân tích kinh nghiệm từ CV Agent]
  - Học vấn (85%): [Phân tích học vấn từ CV Agent]
- **Điểm mạnh:**
  + [Strengths từ CV Agent - hiển thị đầy đủ]
- **Điểm cần cải thiện:**
  - [Weaknesses từ CV Agent - hiển thị đầy đủ]

**🎯 Data Analyst**
- **Điểm số:** 🟡 65%
- **Phân tích:** [Phân tích chi tiết từ CV Agent]
- **Phân tích chi tiết:**
  - Chức danh (70%): [Phân tích từ CV Agent]
  - Kỹ năng (75%): [Phân tích từ CV Agent]
  - Kinh nghiệm (60%): [Phân tích từ CV Agent]
  - Học vấn (55%): [Phân tích từ CV Agent]
- **Điểm mạnh:**
  + [Strengths từ CV Agent]
- **Điểm cần cải thiện:**
  - [Weaknesses từ CV Agent]

**🎯 Software Engineer**
- **Điểm số:** 🔴 45%
- **Phân tích:** [Phân tích chi tiết từ CV Agent]
[Hiển thị đầy đủ tất cả vị trí với chi tiết từng tiêu chí]"

Trả lời bằng tiếng Việt, sử dụng Markdown để định dạng đẹp.
""".format(
                user_input=user_input,
                query_agent_answer=query_agent_answer or "Không có kết quả từ QueryAgent",
                cv_agent_answer=cv_agent_answer or "Không có kết quả từ CV Agent",
                results_summary=json.dumps(results_summary, ensure_ascii=False, indent=2),
                table_summary=table_summary
            )
            
            response = await llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return f"Không thể thực hiện phân tích AI: {str(e)}"

    async def _final_answer(self, user_input: str, table_markdown: str | None) -> str:
        """Sinh câu trả lời tự nhiên, trực tiếp cho người dùng."""
        try:
            # Kiểm tra AI có enabled không
            if not self.ai_enabled:
                return "AI analysis is disabled (missing API key or dependencies)"
            
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
        Tổng hợp và trình bày kết quả từ các agent khác theo format đẹp
        """
        try:
            print(f"🧠 Analysis Agent: Tổng hợp kết quả cho '{user_input}'")
            print(f"🧠 Analysis Agent: Số lượng agent results: {len(agent_results) if agent_results else 0}")
            print(f"🧠 Analysis Agent: Agent results: {agent_results}")
            
            if not agent_results:
                return {
                    "agent": "analysis_agent",
                    "status": "info",
                    "result": {
                        "message": "Analysis Agent sẵn sàng tổng hợp kết quả",
                        "usage": "Cung cấp kết quả từ các agent khác để tổng hợp và trình bày",
                        "capabilities": [
                            "Tổng hợp kết quả từ Query Agent, CV Agent, Chart Agent",
                            "Trình bày kết quả theo format đẹp mắt với emoji",
                            "Tạo báo cáo tổng hợp chi tiết",
                            "Phân tích và hiển thị key findings"
                        ]
                    }
                }
            
            # Debug: In chi tiết từng agent result
            print(f"🧠 Analysis Agent: Nhận được {len(agent_results)} kết quả từ orchestrator")
            for i, result in enumerate(agent_results):
                print(f"  Result {i}: agent={result.get('agent')}, status={result.get('status')}")
                print(f"    Result type: {type(result)}")
                print(f"    Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")
                
                if result.get('agent') == 'cv_agent':
                    cv_result = result.get('result', {})
                    print(f"    CV Agent result type: {type(cv_result)}")
                    print(f"    CV Agent result keys: {list(cv_result.keys()) if isinstance(cv_result, dict) else 'Not dict'}")
                    
                    # Kiểm tra nếu cv_result có cấu trúc nested (từ orchestrator)
                    if isinstance(cv_result, dict) and 'result' in cv_result:
                        print(f"    CV Agent có nested result structure")
                        nested_result = cv_result.get('result', {})
                        print(f"    Nested result keys: {list(nested_result.keys()) if isinstance(nested_result, dict) else 'Not dict'}")
                        
                        if isinstance(nested_result, dict) and 'cv_evaluations' in nested_result:
                            cv_count = len(nested_result['cv_evaluations'])
                            print(f"    CV evaluations count (nested): {cv_count}")
                            for j, evaluation in enumerate(nested_result['cv_evaluations']):
                                cv_name = evaluation.get('cv_name', f'CV_{j}')
                                print(f"      CV {j+1}: {cv_name}")
                        else:
                            print(f"    Nested result không có cv_evaluations")
                    elif isinstance(cv_result, dict) and 'cv_evaluations' in cv_result:
                        cv_count = len(cv_result['cv_evaluations'])
                        print(f"    CV evaluations count: {cv_count}")
                        for j, evaluation in enumerate(cv_result['cv_evaluations']):
                            cv_name = evaluation.get('cv_name', f'CV_{j}')
                            print(f"      CV {j+1}: {cv_name}")
                    else:
                        print(f"    CV Agent result không có cv_evaluations hoặc không phải dict")
                        print(f"    CV Agent result content: {cv_result}")
            
            # Trích xuất và phân loại kết quả từ các agent
            extracted_results = self._extract_agent_results(agent_results)
            print(f"🧠 Analysis Agent: Extracted results: {extracted_results}")
            
            # Tạo báo cáo tổng hợp với format đẹp
            summary_report = self._create_summary_report(extracted_results, user_input)
            
            # Bỏ qua AI analysis để tiết kiệm token - chỉ trả về markdown format
            ai_analysis = ""
            
            # Tạo markdown summary đẹp mắt (đã có đầy đủ thông tin)
            markdown_summary = summary_report.get("formatted_summary", "")
            print(f"🧠 Analysis Agent: Markdown summary length: {len(markdown_summary)}")
            print(f"🧠 Analysis Agent: Markdown preview: {markdown_summary[:200]}...")

            return {
                "agent": "analysis_agent",
                "status": "success",
                "result": {
                    "formatted_summary": markdown_summary,  # Ưu tiên markdown format
                    "summary_report": summary_report,
                    "ai_analysis": ai_analysis,  # Bỏ trống để tiết kiệm token
                    "key_findings": summary_report.get("key_findings", []),
                    "execution_stats": {
                        "total_agents": summary_report["execution_summary"]["total_agents"],
                        "successful_agents": summary_report["execution_summary"]["successful_agents"],
                        "success_rate": summary_report["execution_summary"]["success_rate"]
                    },
                    "agent_summaries": {
                        agent_name: result.get("summary", "") 
                        for agent_name, result in summary_report.get("agent_results", {}).items()
                    }
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

