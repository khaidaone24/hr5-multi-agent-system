import asyncio
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Warning: PyMuPDF not installed")
    fitz = None
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import time
from langchain_google_genai import ChatGoogleGenerativeAI

class CVAgent:
    """
    CV Agent - Phân tích CV và ứng viên với MCP Server
    """
    
    def __init__(self):
        load_dotenv()
        self.GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not self.GEMINI_API_KEY:
            raise ValueError(" Thiếu GOOGLE_API_KEY trong .env")
        
        # Cấu hình Gemini
        genai.configure(api_key=self.GEMINI_API_KEY)
        
        # LLM cho agent
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash-lite",
            google_api_key=self.GEMINI_API_KEY,
            temperature=0.2,
        )
        
        # Thư mục CV mặc định
        self.cv_folder = Path("D:/HR4/PDF")
        self.job_file = Path("D:/HR4/job_requirements/job_requirements.xlsx")
        
        # Cache cho quota management
        self._quota_tracker = {"minute": 0, "count": 0}
        self._max_requests_per_minute = 15
    
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
    
    def extract_pdf_with_content(self, pdf_path: str) -> Dict[str, Any]:
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
    
    def extract_key_info(self, cv_text: str) -> Dict[str, Any]:
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
    
    def extract_cvs_from_folder(self, folder_path: str) -> Dict[str, Any]:
        """Trích xuất tất cả CV PDF từ thư mục"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"CV folder not found: {folder_path}")
        
        cv_data = {}
        pdf_files = list(folder_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No CV PDF files found in {folder_path}")
            return {}
        
        print(f"Found {len(pdf_files)} CV PDF files in {folder_path}")
        for pdf_file in pdf_files:
            try:
                print(f"Processing CV: {pdf_file.name}")
                cv_info = self.extract_pdf_with_content(str(pdf_file))
                cv_info["key_info"] = self.extract_key_info(cv_info["all_text"])
                cv_data[pdf_file.name] = cv_info
            except Exception as e:
                cv_data[pdf_file.name] = {"error": str(e)}
        
        return cv_data
    
    def compare_cv_job_with_gemini(self, cv_text: str, job_text: str, cv_key_info: Optional[Dict] = None) -> tuple:
        """So sánh CV với yêu cầu công việc bằng Gemini AI"""
        
        # Prepare structured prompt
        key_info_str = ""
        if cv_key_info:
            key_info_str = f"""
KEY CANDIDATE INFO:
- Experience: {cv_key_info.get('experience_years', 'Unknown')}
- Skills: {', '.join(cv_key_info.get('skills', [])[:10])}
- Education: {', '.join(cv_key_info.get('education', []))}
- Contact: {', '.join(cv_key_info.get('emails', []))}
"""
        
        prompt = f"""You are an expert HR recruiter. Analyze this candidate against the job requirements.

{key_info_str}

=== FULL CV CONTENT ===
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
        
        try:
            # Check quota
            quota = self._check_quota()
            if not quota["available"]:
                return 0, quota["message"]
            
            self._increment_quota()
            
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
            
            # Check if blocked
            if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                if response.prompt_feedback.block_reason:
                    return 0, f"Content blocked by safety filter"
            
            result_text = response.text
            
            # Clean markdown
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            return result.get("match_score", 0), result.get("summary", "")
            
        except Exception as e:
            error_msg = str(e)
            
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f" Rate limit hit! Waiting 60 seconds...")
                time.sleep(60)
                return 0, f"Rate limit exceeded. Please wait and try again."
            else:
                print(f" Gemini error: {error_msg[:100]}")
                return 0, f"API Error: {error_msg[:100]}"
    
    async def process(self, user_input: str) -> Dict[str, Any]:
        """
        Xử lý yêu cầu phân tích CV
        """
        try:
            print(f" CV Agent: Xử lý yêu cầu '{user_input}'")
            
            # Phân tích intent cụ thể
            if "so sánh" in user_input.lower() or "compare" in user_input.lower():
                return await self._compare_cvs_with_jobs()
            elif "phân tích" in user_input.lower() or "analyze" in user_input.lower():
                return await self._analyze_cvs()
            elif "tìm ứng viên" in user_input.lower() or "find candidate" in user_input.lower():
                return await self._find_candidates(user_input)
            else:
                # Mặc định: phân tích tất cả CV
                return await self._analyze_cvs()
                
        except Exception as e:
            return {
                "agent": "cv_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def _analyze_cvs(self) -> Dict[str, Any]:
        """Phân tích tất cả CV trong thư mục"""
        try:
            print("🔄 CV Agent: Đang tải dữ liệu CV...")
            cv_data = self.extract_cvs_from_folder(str(self.cv_folder))
            
            if not cv_data:
                return {
                    "agent": "cv_agent",
                    "status": "success",
                    "result": {
                        "message": "Không tìm thấy CV nào trong thư mục",
                        "cv_count": 0
                    }
                }
            
            # Tạo báo cáo phân tích
            analysis_result = {
                "total_cvs": len(cv_data),
                "cv_summaries": [],
                "skills_analysis": {},
                "experience_analysis": {}
            }
            
            all_skills = []
            all_experience = []
            
            for cv_name, cv_info in cv_data.items():
                if "error" in cv_info:
                    continue
                
                key_info = cv_info.get("key_info", {})
                skills = key_info.get("skills", [])
                experience = key_info.get("experience_years", "Unknown")
                
                all_skills.extend(skills)
                if experience != "Unknown":
                    all_experience.append(experience)
                
                analysis_result["cv_summaries"].append({
                    "cv_name": cv_name,
                    "skills": skills,
                    "experience": experience,
                    "education": key_info.get("education", []),
                    "contact": {
                        "emails": key_info.get("emails", []),
                        "phones": key_info.get("phones", [])
                    }
                })
            
            # Phân tích kỹ năng
            from collections import Counter
            skills_counter = Counter(all_skills)
            analysis_result["skills_analysis"] = dict(skills_counter.most_common(10))
            
            # Phân tích kinh nghiệm
            analysis_result["experience_analysis"] = {
                "total_with_experience": len(all_experience),
                "experience_distribution": dict(Counter(all_experience))
            }
            
            return {
                "agent": "cv_agent",
                "status": "success",
                "result": analysis_result
            }
            
        except Exception as e:
            return {
                "agent": "cv_agent",
                "status": "error",
                "error": str(e)
            }
    
    async def _compare_cvs_with_jobs(self) -> Dict[str, Any]:
        """So sánh CV với yêu cầu công việc"""
        try:
            print("🔄 CV Agent: Đang tải dữ liệu CV và Job requirements...")
            
            # Load CV data
            cv_data = self.extract_cvs_from_folder(str(self.cv_folder))
            
            # Load job requirements
            job_data = {}
            if self.job_file.exists():
                df = pd.read_excel(self.job_file)
                for idx, row in df.iterrows():
                    job_name = row.get('Job Title', f'Job_{idx}')
                    job_data[job_name] = row.to_dict()
            
            if not job_data:
                return {
                    "agent": "cv_agent",
                    "status": "success",
                    "result": {
                        "message": "Không tìm thấy job requirements",
                        "cv_count": len(cv_data)
                    }
                }
            
            # So sánh từng CV với từng job
            match_results = {}
            total_pairs = len(job_data) * len(cv_data)
            current = 0
            
            for job_name, job_info in job_data.items():
                job_text = " ".join(str(v) for k, v in job_info.items())
                for cv_name, cv_info in cv_data.items():
                    current += 1
                    print(f"  [{current}/{total_pairs}] {job_name[:25]}... ↔ {cv_name[:25]}...")
                    
                    if "error" in cv_info:
                        score, summary = 0, "CV extraction error"
                    else:
                        cv_text = cv_info.get("all_text", "")
                        key_info = cv_info.get("key_info", {})
                        
                        if not cv_text:
                            score, summary = 0, "Empty CV"
                        else:
                            score, summary = self.compare_cv_job_with_gemini(cv_text, job_text, key_info)
                            time.sleep(2)  # Delay để tránh rate limit
                    
                    key = f"{job_name} -> {cv_name}"
                    match_results[key] = {
                        "match_percentage": score,
                        "summary": summary
                    }
                    print(f"    ✓ Score: {score}/100")
            
            return {
                "agent": "cv_agent",
                "status": "success",
                "result": {
                    "total_cvs": len(cv_data),
                    "total_jobs": len(job_data),
                    "match_results": match_results
                }
            }
            
        except Exception as e:
            return {
                "agent": "cv_agent",
                "status": "error",
                "error": str(e)
            }
    
    async def _find_candidates(self, user_input: str) -> Dict[str, Any]:
        """Tìm ứng viên phù hợp dựa trên yêu cầu"""
        try:
            print(" CV Agent: Tìm kiếm ứng viên phù hợp...")
            
            # Load CV data
            cv_data = self.extract_cvs_from_folder(str(self.cv_folder))
            
            if not cv_data:
                return {
                    "agent": "cv_agent",
                    "status": "success",
                    "result": {
                        "message": "Không tìm thấy CV nào",
                        "candidates": []
                    }
                }
            
            # Tìm kiếm dựa trên từ khóa trong user_input
            search_keywords = user_input.lower().split()
            candidates = []
            
            for cv_name, cv_info in cv_data.items():
                if "error" in cv_info:
                    continue
                
                cv_text = cv_info.get("all_text", "").lower()
                key_info = cv_info.get("key_info", {})
                skills = key_info.get("skills", [])
                
                # Tính điểm phù hợp
                match_score = 0
                matched_keywords = []
                
                for keyword in search_keywords:
                    if keyword in cv_text:
                        match_score += 1
                        matched_keywords.append(keyword)
                    elif any(keyword in skill.lower() for skill in skills):
                        match_score += 2  # Skills match có trọng số cao hơn
                        matched_keywords.append(f"skill:{keyword}")
                
                if match_score > 0:
                    candidates.append({
                        "cv_name": cv_name,
                        "match_score": match_score,
                        "matched_keywords": matched_keywords,
                        "skills": skills,
                        "experience": key_info.get("experience_years", "Unknown"),
                        "summary": f"Phù hợp với {len(matched_keywords)} tiêu chí"
                    })
            
            # Sắp xếp theo điểm phù hợp
            candidates.sort(key=lambda x: x["match_score"], reverse=True)
            
            return {
                "agent": "cv_agent",
                "status": "success",
                "result": {
                    "search_query": user_input,
                    "total_candidates": len(candidates),
                    "top_candidates": candidates[:5]  # Top 5 ứng viên
                }
            }
            
        except Exception as e:
            return {
                "agent": "cv_agent",
                "status": "error",
                "error": str(e)
            }

# Test function
async def test_cv_agent():
    agent = CVAgent()
    
    test_cases = [
        "Phân tích tất cả CV",
        "So sánh CV với yêu cầu công việc",
        "Tìm ứng viên có kinh nghiệm Python",
        "Tìm ứng viên có bằng đại học"
    ]
    
    for test_input in test_cases:
        print(f"\n{'='*50}")
        print(f"Test: {test_input}")
        result = await agent.process(test_input)
        print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_cv_agent())
