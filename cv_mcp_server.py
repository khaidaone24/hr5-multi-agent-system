#!/usr/bin/env python3
"""
MCP Server cho CV Agent
Cung cấp các tools để phân tích CV và ứng viên
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import fitz  # PyMuPDF
import pandas as pd
import re
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    ListToolsRequest,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class CVMCPServer:
    def __init__(self):
        self.server = Server("cv-analysis-server")
        self.cv_folder = Path("D:/HR4/PDF")
        self.job_file = Path("D:/HR4/job_requirements/job_requirements.xlsx")
        self._quota_tracker = {"minute": 0, "count": 0}
        self._max_requests_per_minute = 15
        
        # Register tools
        self._register_tools()
    
    def _check_quota(self) -> Dict[str, Any]:
        """Kiểm tra quota Gemini"""
        current_minute = int(time.time() / 60)
        
        if self._quota_tracker.get("minute") != current_minute:
            self._quota_tracker = {"minute": current_minute, "count": 0}
        
        if self._quota_tracker["count"] >= self._max_requests_per_minute:
            wait_seconds = 60 - (time.time() % 60)
            return {
                "available": False,
                "wait_seconds": wait_seconds,
                "message": f"Quota exceeded. Wait {wait_seconds:.0f}s"
            }
        
        return {"available": True}
    
    def _increment_quota(self):
        """Tăng quota counter"""
        self._quota_tracker["count"] = self._quota_tracker.get("count", 0) + 1
    
    def _register_tools(self):
        """Đăng ký các tools"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="extract_cv",
                    description="Trích xuất và phân tích CV từ file PDF",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pdf_path": {
                                "type": "string",
                                "description": "Đường dẫn đến file PDF CV"
                            }
                        },
                        "required": ["pdf_path"]
                    }
                ),
                Tool(
                    name="extract_cvs_from_folder",
                    description="Trích xuất tất cả CV từ thư mục",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "folder_path": {
                                "type": "string",
                                "description": "Đường dẫn đến thư mục chứa CV"
                            }
                        },
                        "required": ["folder_path"]
                    }
                ),
                Tool(
                    name="compare_cv_job",
                    description="So sánh CV với yêu cầu công việc",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "cv_text": {
                                "type": "string",
                                "description": "Nội dung CV"
                            },
                            "job_text": {
                                "type": "string", 
                                "description": "Yêu cầu công việc"
                            }
                        },
                        "required": ["cv_text", "job_text"]
                    }
                ),
                Tool(
                    name="analyze_cv_skills",
                    description="Phân tích kỹ năng từ CV",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "cv_text": {
                                "type": "string",
                                "description": "Nội dung CV"
                            }
                        },
                        "required": ["cv_text"]
                    }
                ),
                Tool(
                    name="find_candidates",
                    description="Tìm ứng viên phù hợp dựa trên tiêu chí",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "criteria": {
                                "type": "string",
                                "description": "Tiêu chí tìm kiếm ứng viên"
                            },
                            "folder_path": {
                                "type": "string",
                                "description": "Đường dẫn thư mục CV"
                            }
                        },
                        "required": ["criteria"]
                    }
                ),
                Tool(
                    name="get_job_requirements",
                    description="Lấy danh sách yêu cầu công việc",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            try:
                if name == "extract_cv":
                    return await self._extract_cv(arguments)
                elif name == "extract_cvs_from_folder":
                    return await self._extract_cvs_from_folder(arguments)
                elif name == "compare_cv_job":
                    return await self._compare_cv_job(arguments)
                elif name == "analyze_cv_skills":
                    return await self._analyze_cv_skills(arguments)
                elif name == "find_candidates":
                    return await self._find_candidates(arguments)
                elif name == "get_job_requirements":
                    return await self._get_job_requirements(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _extract_pdf_with_content(self, pdf_path: str) -> Dict[str, Any]:
        """Trích xuất nội dung PDF với cấu trúc"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        pdf_data = {"headings": {}, "all_text": "", "structured_data": {}}
        current_heading = None
        
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" not in b:
                    continue
                for line in b["lines"]:
                    line_text = ""
                    max_font = 0
                    bold_count = 0
                    for span in line["spans"]:
                        line_text += span["text"] + " "
                        max_font = max(max_font, span["size"])
                        if span.get("flags", 0) & 2**4:
                            bold_count += 1
                    line_text = line_text.strip()
                    if not line_text:
                        continue
                    pdf_data["all_text"] += line_text + "\n"
                    
                    # Heading detection
                    is_heading = max_font > 12 or bold_count > 0 or line_text.isupper() or line_text.endswith(':')
                    if is_heading:
                        heading_key = f"{line_text} (page {page_num})"
                        pdf_data["headings"][heading_key] = {
                            "font_size": max_font,
                            "bold_count": bold_count,
                            "page": page_num,
                            "text": line_text,
                            "content": ""
                        }
                        current_heading = heading_key
                    else:
                        if current_heading:
                            pdf_data["headings"][current_heading]["content"] += line_text + "\n"
        
        return pdf_data
    
    def _extract_key_info(self, cv_text: str) -> Dict[str, Any]:
        """Trích xuất thông tin quan trọng từ CV"""
        info = {
            "skills": [],
            "experience_years": "Unknown",
            "education": [],
            "emails": [],
            "phones": []
        }
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', cv_text)
        info["emails"] = emails[:2]
        
        # Extract phones
        phones = re.findall(r'\b(?:\+84|0)\d{9,10}\b', cv_text)
        info["phones"] = phones[:2]
        
        # Look for years of experience
        exp_match = re.search(r'(\d+)\+?\s*(?:years?|năm|YoE)', cv_text, re.IGNORECASE)
        if exp_match:
            info["experience_years"] = exp_match.group(1) + " years"
        
        # Extract education keywords
        edu_keywords = ['university', 'college', 'bachelor', 'master', 'phd', 'degree', 'đại học', 'cao đẳng']
        for keyword in edu_keywords:
            if keyword.lower() in cv_text.lower():
                info["education"].append(keyword.capitalize())
        
        # Extract common skills
        skill_keywords = ['python', 'java', 'javascript', 'react', 'sql', 'aws', 'docker', 
                          'machine learning', 'data analysis', 'project management', 'agile',
                          'communication', 'leadership', 'teamwork']
        for skill in skill_keywords:
            if skill.lower() in cv_text.lower():
                info["skills"].append(skill)
        
        return info
    
    async def _extract_cv(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Trích xuất CV từ file PDF"""
        pdf_path = arguments.get("pdf_path")
        if not pdf_path:
            return [TextContent(type="text", text="Error: pdf_path is required")]
        
        try:
            cv_data = self._extract_pdf_with_content(pdf_path)
            cv_data["key_info"] = self._extract_key_info(cv_data["all_text"])
            
            result = {
                "success": True,
                "cv_data": cv_data,
                "summary": {
                    "total_text_length": len(cv_data["all_text"]),
                    "headings_count": len(cv_data["headings"]),
                    "key_info": cv_data["key_info"]
                }
            }
            
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error extracting CV: {str(e)}")]
    
    async def _extract_cvs_from_folder(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Trích xuất tất cả CV từ thư mục"""
        folder_path = arguments.get("folder_path", str(self.cv_folder))
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return [TextContent(type="text", text=f"Error: Folder not found: {folder_path}")]
        
        try:
            cv_data = {}
            pdf_files = list(folder_path.glob("*.pdf"))
            
            if not pdf_files:
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "message": f"No PDF files found in {folder_path}",
                    "cv_data": {}
                }, ensure_ascii=False))]
            
            for pdf_file in pdf_files:
                try:
                    cv_info = self._extract_pdf_with_content(str(pdf_file))
                    cv_info["key_info"] = self._extract_key_info(cv_info["all_text"])
                    cv_data[pdf_file.name] = cv_info
                except Exception as e:
                    cv_data[pdf_file.name] = {"error": str(e)}
            
            result = {
                "success": True,
                "total_cvs": len(cv_data),
                "cv_data": cv_data
            }
            
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error processing folder: {str(e)}")]
    
    async def _compare_cv_job(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """So sánh CV với yêu cầu công việc"""
        cv_text = arguments.get("cv_text", "")
        job_text = arguments.get("job_text", "")
        
        if not cv_text or not job_text:
            return [TextContent(type="text", text="Error: cv_text and job_text are required")]
        
        try:
            # Check quota
            quota = self._check_quota()
            if not quota["available"]:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": quota["message"]
                }, ensure_ascii=False))]
            
            self._increment_quota()
            
            # Prepare prompt for Gemini
            prompt = f"""You are an expert HR recruiter. Analyze this candidate against the job requirements.

=== CV CONTENT ===
{cv_text[:8000]}

=== JOB REQUIREMENTS ===
{job_text[:2000]}

Evaluate the match (0-100) based on:
1. Technical/functional skills alignment
2. Years of experience match
3. Educational qualification
4. Industry/domain experience
5. Cultural fit indicators

Return ONLY this JSON format:
{{
    "match_score": <integer 0-100>,
    "summary": "<2-3 sentences explaining the score>"
}}
"""
            
            # Configure safety settings
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500,
                    top_p=0.8,
                ),
                safety_settings=safety_settings
            )
            
            result_text = response.text
            
            # Clean markdown
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "match_score": result.get("match_score", 0),
                "summary": result.get("summary", ""),
                "quota_used": self._quota_tracker["count"]
            }, ensure_ascii=False))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error comparing CV: {str(e)}")]
    
    async def _analyze_cv_skills(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Phân tích kỹ năng từ CV"""
        cv_text = arguments.get("cv_text", "")
        
        if not cv_text:
            return [TextContent(type="text", text="Error: cv_text is required")]
        
        try:
            key_info = self._extract_key_info(cv_text)
            
            result = {
                "success": True,
                "skills_analysis": {
                    "skills": key_info["skills"],
                    "experience_years": key_info["experience_years"],
                    "education": key_info["education"],
                    "contact": {
                        "emails": key_info["emails"],
                        "phones": key_info["phones"]
                    }
                }
            }
            
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error analyzing skills: {str(e)}")]
    
    async def _find_candidates(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Tìm ứng viên phù hợp"""
        criteria = arguments.get("criteria", "")
        folder_path = arguments.get("folder_path", str(self.cv_folder))
        
        if not criteria:
            return [TextContent(type="text", text="Error: criteria is required")]
        
        try:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                return [TextContent(type="text", text=f"Error: Folder not found: {folder_path}")]
            
            # Load CVs
            cv_data = {}
            pdf_files = list(folder_path.glob("*.pdf"))
            
            for pdf_file in pdf_files:
                try:
                    cv_info = self._extract_pdf_with_content(str(pdf_file))
                    cv_info["key_info"] = self._extract_key_info(cv_info["all_text"])
                    cv_data[pdf_file.name] = cv_info
                except Exception as e:
                    cv_data[pdf_file.name] = {"error": str(e)}
            
            # Search based on criteria
            search_keywords = criteria.lower().split()
            candidates = []
            
            for cv_name, cv_info in cv_data.items():
                if "error" in cv_info:
                    continue
                
                cv_text = cv_info.get("all_text", "").lower()
                key_info = cv_info.get("key_info", {})
                skills = key_info.get("skills", [])
                
                # Calculate match score
                match_score = 0
                matched_keywords = []
                
                for keyword in search_keywords:
                    if keyword in cv_text:
                        match_score += 1
                        matched_keywords.append(keyword)
                    elif any(keyword in skill.lower() for skill in skills):
                        match_score += 2
                        matched_keywords.append(f"skill:{keyword}")
                
                if match_score > 0:
                    candidates.append({
                        "cv_name": cv_name,
                        "match_score": match_score,
                        "matched_keywords": matched_keywords,
                        "skills": skills,
                        "experience": key_info.get("experience_years", "Unknown")
                    })
            
            # Sort by match score
            candidates.sort(key=lambda x: x["match_score"], reverse=True)
            
            result = {
                "success": True,
                "search_criteria": criteria,
                "total_candidates": len(candidates),
                "top_candidates": candidates[:5]
            }
            
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error finding candidates: {str(e)}")]
    
    async def _get_job_requirements(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Lấy danh sách yêu cầu công việc"""
        try:
            if not self.job_file.exists():
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "message": "No job requirements file found",
                    "job_data": {}
                }, ensure_ascii=False))]
            
            df = pd.read_excel(self.job_file)
            job_data = {}
            for idx, row in df.iterrows():
                job_name = row.get('Job Title', f'Job_{idx}')
                job_data[job_name] = row.to_dict()
            
            result = {
                "success": True,
                "total_jobs": len(job_data),
                "job_data": job_data
            }
            
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting job requirements: {str(e)}")]
    
    async def run(self):
        """Chạy MCP Server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="cv-analysis-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None,
                    ),
                ),
            )

async def main():
    """Main function"""
    server = CVMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
