import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalysisAgent:
    """Refactored Analysis Agent với code structure tốt hơn"""
    
    def __init__(self):
        self.agent_name = "analysis_agent"
        self.llm_model = "models/gemini-2.5-flash-lite"
        load_dotenv()
        self.GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
        logger.info("Analysis Agent initialized with refactored structure")
    
    def _format_cv_evaluation(self, evaluation: Dict[str, Any], index: int) -> List[str]:
        """Format một CV evaluation thành markdown lines"""
        lines = []
        cv_name = evaluation.get("cv_name", f"CV_{index}")
        status = evaluation.get("status", "Unknown")
        
        # Header
        lines.append(f"**👤 Ứng viên {index}: {cv_name}**")
        lines.append(f"- **Trạng thái:** {status}")
        
        # Xử lý lỗi 429
        if status == "error":
            error_msg = evaluation.get("error", "")
            if "429" in error_msg or "Rate limit" in error_msg:
                lines.extend(self._format_rate_limit_error(error_msg))
                return lines
        
        # CV info
        cv_key_info = evaluation.get("cv_key_info", {})
        if cv_key_info.get("experience_years"):
            lines.append(f"- **Kinh nghiệm:** {cv_key_info['experience_years']} năm")
        if cv_key_info.get("skills"):
            skills = ', '.join(cv_key_info['skills'][:5])
            lines.append(f"- **Kỹ năng:** {skills}")
        lines.append("")
        
        # Best match
        if evaluation.get("best_match"):
            lines.extend(self._format_best_match(evaluation["best_match"], cv_name))
        
        # All evaluations
        if evaluation.get("all_evaluations"):
            lines.extend(self._format_all_evaluations(evaluation["all_evaluations"]))
        
        lines.append("---\n")
        return lines
    
    def _format_best_match(self, best_match: Dict[str, Any], cv_name: str) -> List[str]:
        """Format best match evaluation"""
        lines = []
        score = best_match.get("score", 0)
        score_icon = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
        
        lines.append("**Xếp hạng AI (beta)**")
        lines.append(f"**Điểm: {score_icon} {score}%**\n")
        
        # Detailed scores
        if best_match.get("detailed_scores"):
            lines.append("**Phân tích chi tiết:**")
            for criteria, data in best_match["detailed_scores"].items():
                criteria_name = self._get_criteria_name(criteria)
                score = data.get("score", 0)
                analysis = data.get("analysis", "")
                lines.append(f"- **{criteria_name} ({score}%):** {analysis}")
            lines.append("")
        
        # Strengths & Weaknesses
        if best_match.get("strengths"):
            lines.append("**Điểm mạnh:**")
            lines.extend([f"- {s}" for s in best_match["strengths"]])
            lines.append("")
        
        if best_match.get("weaknesses"):
            lines.append("**Điểm cần cải thiện:**")
            lines.extend([f"- {w}" for w in best_match["weaknesses"]])
            lines.append("")
        
        # Chart
        lines.extend(self._create_donut_chart_section(score, cv_name))
        
        return lines
    
    def _format_all_evaluations(self, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Format all job evaluations"""
        lines = ["**CHI TIẾT TẤT CẢ VỊ TRÍ:**\n"]
        
        for eval_item in evaluations:
            job = eval_item.get("job_title", "Unknown")
            score = eval_item.get("score", 0)
            analysis = eval_item.get("analysis", "")
            icon = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
            
            lines.append(f"**{job}**")
            lines.append(f"- **Điểm:** {icon} {score}%")
            lines.append(f"- **Phân tích:** {analysis}")
            
            # Detailed scores
            if eval_item.get("detailed_scores"):
                lines.append("- **Chi tiết:**")
                for criteria, data in eval_item["detailed_scores"].items():
                    name = self._get_criteria_name(criteria)
                    s = data.get("score", 0)
                    a = data.get("analysis", "")
                    lines.append(f"  - {name} ({s}%): {a}")
            
            # Strengths/Weaknesses
            if eval_item.get("strengths"):
                lines.append("- **Điểm mạnh:**")
                lines.extend([f"  + {s}" for s in eval_item["strengths"]])
            
            if eval_item.get("weaknesses"):
                lines.append("- **Cần cải thiện:**")
                lines.extend([f"  - {w}" for w in eval_item["weaknesses"]])
            
            # Thêm donut chart cho từng vị trí
            lines.extend(self._create_donut_chart_section(score, f"{job}"))
            
            lines.append("\n---\n")
        
        return lines
    
    def _get_criteria_name(self, criteria: str) -> str:
        """Get Vietnamese name for criteria"""
        mapping = {
            "job_title": "Chức danh",
            "skills": "Kỹ năng",
            "experience": "Kinh nghiệm",
            "education": "Học vấn"
        }
        return mapping.get(criteria, criteria)
    
    def _format_rate_limit_error(self, error_msg: str) -> List[str]:
        """Format rate limit error message"""
        return [
            "",
            "🚨" * 20,
            "**LỖI RATE LIMIT 429**",
            f"**Lỗi:** {error_msg}",
            "⏰ **Thời gian:** " + datetime.now().strftime('%H:%M:%S'),
            "🛑 **Hệ thống đã dừng để tránh lỗi API**",
            "💡 **Giải pháp:** Thử lại sau 1-2 phút",
            "🤖 **Model:** Gemini 2.5 Flash Lite",
            "🚨" * 20,
            ""
        ]
    
    def _create_donut_chart_section(self, score: int, cv_name: str) -> List[str]:
        """Create donut chart section"""
        lines = ["**Biểu đồ đánh giá:**"]
        
        try:
            from chart_agent import ChartAgent
            chart_agent = ChartAgent()
            logger.info(f"Creating donut chart for score: {score}, cv_name: {cv_name}")
            
            # Rút ngắn title để tránh chart quá rộng
            short_title = f"Đánh Giá: {score}%"
            result = chart_agent._create_donut_chart(
                score, 
                100 - score, 
                short_title
            )
            
            logger.info(f"Donut chart result: {result}")
            
            if "chart_file" in result:
                # Convert path to API route
                chart_filename = result.get('chart_filename', '')
                api_path = f"/api/charts/{chart_filename}"
                lines.append(f"![Donut Chart]({api_path})")
                lines.append(f"*{result.get('title', 'Đánh giá phù hợp')}*")
                logger.info(f"Donut chart created successfully: {api_path}")
            else:
                logger.warning("No chart_file in result, using text chart")
                lines.extend(self._create_text_chart(score))
        except Exception as e:
            logger.error(f"Failed to create donut chart: {e}")
            lines.extend(self._create_text_chart(score))
        
        lines.append("")
        return lines
    
    def _create_text_chart(self, score: int) -> List[str]:
        """Fallback text chart"""
        return [
            "```",
            f"🔴 Phù hợp: {score}%",
            f"🟢 Không phù hợp: {100 - score}%",
            "```"
        ]
    
    def _extract_cv_evaluations(self, cv_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract CV evaluations from nested structure"""
        # Check nested result structure
        if isinstance(cv_result, dict) and "result" in cv_result:
            nested = cv_result.get("result", {})
            if isinstance(nested, dict) and "cv_evaluations" in nested:
                return nested["cv_evaluations"]
        
        # Check direct structure
        if isinstance(cv_result, dict) and "cv_evaluations" in cv_result:
            return cv_result["cv_evaluations"]
        
        return []
    
    def _create_formatted_summary(self, agent_results: Dict[str, Any], user_input: str) -> str:
        """Tạo summary với code đã được refactor"""
        summary_parts = []
        
        # Header
        summary_parts.extend([
            "## Báo Cáo Tổng Hợp",
            f"**Yêu cầu:** {user_input}",
            f"**Thời gian:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            ""
        ])
        
        # Execution Summary
        successful = len([r for r in agent_results.values() if r and r.get("status") == "success"])
        total = len([r for r in agent_results.values() if r is not None])
        
        summary_parts.extend([
            "### Tóm Tắt Thực Hiện",
            f"- **Tổng số agent:** {total}",
            f"- **Thành công:** {successful}",
            f"- **Tỷ lệ:** {(successful/total*100):.1f}%" if total > 0 else "- **Tỷ lệ:** 0%",
            ""
        ])
        
        # Key Findings
        key_findings = self._generate_key_findings(agent_results)
        if key_findings:
            summary_parts.append("### Phát Hiện Chính")
            summary_parts.extend([f"- {f}" for f in key_findings])
            summary_parts.append("")
        
        # Agent Results
        summary_parts.append("### Kết Quả Chi Tiết")
        for agent_name, result in agent_results.items():
            if result:
                summary_parts.extend(self._format_agent_result(agent_name, result))
        
        return "\n".join(summary_parts)
    
    def _format_agent_result(self, agent_name: str, result: Dict[str, Any]) -> List[str]:
        """Format một agent result"""
        lines = []
        status_icon = {
            "success": "Thành công",
            "error": "Lỗi"
        }.get(result.get("status"), "Không xác định")
        
        agent_display = {
            "query_agent": "Query Agent",
            "cv_agent": "CV Agent",
            "chart_agent": "Chart Agent",
            "analysis_agent": "Analysis Agent"
        }.get(agent_name, agent_name)
        
        lines.append(f"#### {status_icon} {agent_display}")
        lines.append(f"- **Trạng thái:** {self._summarize_agent_result(result)}")
        
        # Metrics
        key_data = self._extract_key_data(result)
        for metric, value in key_data["metrics"].items():
            lines.append(f"- **{metric.replace('_', ' ').title()}:** {value}")
        
        # Files
        if key_data["files_created"]:
            lines.append(f"- **Files:** {', '.join(key_data['files_created'])}")
        
        # CV specific formatting
        if agent_name == "cv_agent" and key_data.get("full_cv_data"):
            lines.append("\n##### Báo Cáo Đánh Giá Ứng Viên")
            for i, evaluation in enumerate(key_data["full_cv_data"], 1):
                lines.extend(self._format_cv_evaluation(evaluation, i))
        
        lines.append("")
        return lines
    
    def _extract_key_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Trích xuất dữ liệu quan trọng từ kết quả agent"""
        key_data = {
            "data_type": "unknown",
            "data_summary": "",
            "metrics": {},
            "files_created": [],
            "full_cv_data": []
        }

        if result.get("status") != "success":
            return key_data

        agent_name = result.get("agent", "unknown")
        result_data = result.get("result", {})

        if agent_name == "cv_agent":
            key_data["data_type"] = "cv_analysis"
            cv_evaluations = self._extract_cv_evaluations(result_data)
            key_data["full_cv_data"] = cv_evaluations
            key_data["data_summary"] = f"Phân tích {len(cv_evaluations)} CV"
            key_data["metrics"] = {
                "cv_count": len(cv_evaluations),
                "successful_analysis": len([e for e in cv_evaluations if e.get("status") == "success"])
            }

        elif agent_name == "chart_agent":
            key_data["data_type"] = "chart_creation"
            key_data["data_summary"] = result_data.get("summary", "Tạo biểu đồ")
            key_data["files_created"] = result_data.get("files_created", [])
            key_data["metrics"] = {
                "charts_created": len(key_data["files_created"])
            }

        elif agent_name == "query_agent":
            key_data["data_type"] = "data_query"
            key_data["data_summary"] = result_data.get("summary", "Truy vấn dữ liệu")
            key_data["metrics"] = {
                "rows_returned": result_data.get("row_count", 0)
            }

        return key_data
    
    def _generate_key_findings(self, agent_results: Dict[str, Any]) -> List[str]:
        """Tạo key findings từ kết quả"""
        findings = []
        
        for agent_name, result in agent_results.items():
            if result and result.get("status") == "success":
                key_data = self._extract_key_data(result)
                
                if agent_name == "cv_agent" and key_data.get("full_cv_data"):
                    cv_count = len(key_data["full_cv_data"])
                    findings.append(f"Đã phân tích {cv_count} CV thành công")
                
                elif agent_name == "chart_agent" and key_data["files_created"]:
                    chart_count = len(key_data["files_created"])
                    findings.append(f"Tạo {chart_count} biểu đồ thành công")
        
        return findings
    
    async def _generate_llm_query_analysis(self, agent_results_dict: Dict[str, Any], user_input: str) -> str:
        """Tạo phân tích thân thiện cho query results bằng LLM"""
        query_result = agent_results_dict.get("query_agent")
        if not query_result or query_result.get("status") != "success":
            return ""
        
        result_data = query_result.get("result", {})
        if not result_data:
            return ""
        
        # Lấy dữ liệu từ query result
        columns = result_data.get("columns", [])
        data = result_data.get("data", [])
        raw_sql = result_data.get("raw_result", "")
        
        if not data or not columns:
            return ""
        
        try:
            # Khởi tạo LLM
            llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                google_api_key=self.GEMINI_API_KEY,
                temperature=0.3,
            )
            
            # Chuẩn bị dữ liệu cho LLM
            data_sample = data[:10] if len(data) > 10 else data  # Chỉ lấy 10 dòng đầu để tiết kiệm token
            
            prompt = f"""
Bạn là một AI Assistant chuyên chuyển đổi dữ liệu SQL thành ngôn ngữ thân thiện với người dùng.

Yêu cầu của người dùng: "{user_input}"
SQL đã thực thi: {raw_sql}
Các cột dữ liệu: {columns}
Dữ liệu trả về: {data_sample}
Tổng số bản ghi: {len(data)}

Hãy tạo một phân tích thân thiện với người dùng theo format markdown:

1. **Tiêu đề phù hợp** với yêu cầu người dùng
2. **Danh sách dữ liệu** được format đẹp mắt với emoji
3. **Thống kê tổng kết** ở cuối
4. **Ngôn ngữ tiếng Việt**, thân thiện, dễ hiểu
5. **Sử dụng emoji** để làm cho nội dung sinh động
6. **Không hiển thị raw data** mà chuyển thành thông tin có ý nghĩa

Ví dụ format:
### 🏢 Danh Sách Phòng Ban

**Phòng Nhân sự** (PB01)
- 📝 Mô tả: Quản lý nhân sự và tuyển dụng  
- 📅 Ngày thành lập: 2010-01-01
- Trạng thái: Đang hoạt động

**Tổng kết:** Có 3 phòng ban đang hoạt động trong hệ thống.

Chỉ trả về markdown, không cần giải thích thêm.
"""
            
            response = await llm.ainvoke(prompt)
            llm_analysis = response.content if hasattr(response, 'content') else str(response)
            
            # Thêm HTML table vào cuối phân tích
            html_table = self._create_html_table(columns, data)
            
            return llm_analysis + "\n\n" + html_table
            
        except Exception as e:
            logger.error(f"Error generating LLM query analysis: {e}")
            # Fallback: chỉ hiển thị HTML table
            html_table = self._create_html_table(columns, data)
            return f"### Kết Quả Truy Vấn\n\nĐã truy vấn thành công. Có {len(data)} bản ghi được trả về.\n\n" + html_table

    def _create_html_table(self, columns: list, data: list) -> str:
        """Tạo bảng HTML đẹp mắt từ dữ liệu"""
        if not columns or not data:
            return ""
        
        # Tạo header
        html = '<div class="table-responsive mt-3">\n'
        html += '<table class="table table-striped table-hover table-bordered">\n'
        html += '<thead class="table-dark">\n<tr>\n'
        
        # Thêm các cột header
        for col in columns:
            html += f'<th scope="col">{col}</th>\n'
        html += '</tr>\n</thead>\n'
        
        # Thêm dữ liệu
        html += '<tbody>\n'
        for row in data:
            html += '<tr>\n'
            for cell in row:
                # Format cell value
                if cell is None:
                    cell_value = '-'
                elif isinstance(cell, (int, float)):
                    # Format số tiền
                    if 'luong' in str(cell).lower() or 'salary' in str(cell).lower():
                        cell_value = f"{cell:,.0f} VNĐ"
                    else:
                        cell_value = f"{cell:,}"
                elif hasattr(cell, 'strftime'):  # datetime object
                    cell_value = cell.strftime('%d/%m/%Y')
                else:
                    cell_value = str(cell)
                
                html += f'<td>{cell_value}</td>\n'
            html += '</tr>\n'
        html += '</tbody>\n'
        html += '</table>\n'
        html += '</div>\n'
        
        # Thêm thông tin tổng số bản ghi
        html += f'<div class="text-muted mt-2"><small>Tổng số bản ghi: {len(data)}</small></div>\n'
        
        return html
    
    def _summarize_agent_result(self, result: Dict[str, Any]) -> str:
        """Tóm tắt kết quả agent"""
        if result.get("status") == "success":
            key_data = self._extract_key_data(result)
            agent_name = result.get("agent", "unknown")
            
            if agent_name == "cv_agent":
                if key_data.get("full_cv_data"):
                    return f"Phân tích CV và ứng viên thành công"
            elif agent_name == "chart_agent":
                if key_data["files_created"]:
                    return f"Tạo biểu đồ thành công: {key_data['data_summary']}"
                else:
                    return "Tạo biểu đồ thành công"
            else:
                return "Xử lý thành công"
        elif result.get("status") == "error":
            return f"Lỗi: {result.get('error', 'Unknown error')}"
        else:
            return "Trạng thái không xác định"
    
    async def process(self, user_input: str, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main processing method"""
        logger.info(f"Analysis Agent: Processing {len(agent_results)} agent results")
        
        # Extract agent results
        agent_results_dict = self._extract_agent_results(agent_results)
        
        # Create summary report
        summary_report = self._create_summary_report(agent_results_dict, user_input)
        
        # Tạo AI analysis cho query results bằng LLM
        ai_analysis = await self._generate_llm_query_analysis(agent_results_dict, user_input)
        
        # Tạo markdown summary đẹp mắt
        markdown_summary = summary_report.get("formatted_summary", "")
        
        # Thêm AI analysis vào markdown nếu có
        if ai_analysis:
            markdown_summary += "\n\n" + ai_analysis

        return {
            "agent": "analysis_agent",
            "status": "success",
            "result": {
                "formatted_summary": markdown_summary,
                "summary_report": summary_report,
                "ai_analysis": ai_analysis,
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
    
    def _extract_agent_results(self, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract agent results from list format"""
        results_dict = {}
        for result in agent_results:
            if result and result.get("agent"):
                agent_name = result["agent"]
                results_dict[agent_name] = result
        return results_dict
    
    def _create_summary_report(self, agent_results: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Tạo summary report"""
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
        
        # Tổng hợp kết quả từng agent
        for agent_name, result in agent_results.items():
            if result:
                report["agent_results"][agent_name] = {
                    "status": result.get("status"),
                    "summary": self._summarize_agent_result(result),
                    "key_data": self._extract_key_data(result)
                }
        
        # Tạo key findings
        report["key_findings"] = self._generate_key_findings(agent_results)
        
        # Tạo formatted summary
        report["formatted_summary"] = self._create_formatted_summary(agent_results, user_input)
        
        return report
