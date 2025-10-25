import asyncio
import logging
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    logger.warning("PyMuPDF not installed. PDF processing will be limited.")
    fitz = None
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import time
from datetime import datetime
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
        self.model_name = "models/gemini-2.5-flash-lite"  # Model chính cho CV analysis
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash-lite",
            google_api_key=self.GEMINI_API_KEY,
            temperature=0.2,
        )
        
        # Thư mục CV mặc định
        self.cv_folder = Path("cvs")
        self.job_file = Path("job_requirements/job_requirements.xlsx")
        
        # Cache cho quota management
        self._quota_tracker = {"minute": 0, "count": 0}
        self._max_requests_per_minute = 15
    
    def _extract_json_from_text(self, text: str) -> str:
        """Trích xuất JSON object từ text"""
        try:
            # Tìm { đầu tiên và } cuối cùng
            start = text.find('{')
            if start == -1:
                return ""
            
            # Đếm braces để tìm } cuối cùng
            brace_count = 0
            end = start
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            return text[start:end]
        except:
            return ""
    
    def _fix_json_quotes(self, text: str) -> str:
        """Sửa quotes trong JSON"""
        try:
            # Thay thế smart quotes bằng regular quotes
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
            
            # Sửa unterminated strings
            text = re.sub(r'"[^"]*$', '"', text)  # Thêm quote cuối cho string chưa kết thúc
            text = re.sub(r'"[^"]*"', lambda m: m.group(0) if m.group(0).count('"') % 2 == 0 else m.group(0) + '"', text)
            
            return text
        except:
            return ""
    
    def _extract_clean_json(self, text: str) -> str:
        """Trích xuất JSON sạch từ text"""
        try:
            # Tìm JSON object hoàn chỉnh
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            if matches:
                # Lấy match dài nhất (có thể là JSON hoàn chỉnh nhất)
                return max(matches, key=len)
            
            return ""
        except:
            return ""
    
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
        print(f" CV Agent: Extracting PDF from: {pdf_path}")
        if not os.path.exists(pdf_path):
            print(f" CV Agent: PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            print(f" CV Agent: PDF opened successfully, {len(doc)} pages")
            pdf_data = {"headings": {}, "all_text": "", "structured_data": {}}
            current_heading = None
        except Exception as e:
            print(f" CV Agent: Error opening PDF: {e}")
            raise
        
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
        
        print(f" CV Agent: PDF extraction completed. Text length: {len(pdf_data['all_text'])}")
        print(f" CV Agent: Text preview: {pdf_data['all_text'][:200]}...")
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
        """So sánh CV với yêu cầu công việc bằng Gemini AI - phân tích chi tiết từng tiêu chí"""
        
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
        
        prompt = f"""Bạn là chuyên gia tuyển dụng HR. Phân tích ứng viên này so với yêu cầu công việc.

{key_info_str}

=== NỘI DUNG CV ĐẦY ĐỦ ===
{cv_text[:8000]}

=== YÊU CẦU CÔNG VIỆC ===
{job_text[:2000]}

Đánh giá mức độ phù hợp (0-100) dựa trên:
1. Sự phù hợp chức danh
2. Kỹ năng phù hợp  
3. Mức độ kinh nghiệm
4. Trình độ học vấn

QUAN TRỌNG: Chỉ trả về JSON hợp lệ. Giữ phân tích ngắn gọn (tối đa 100 ký tự mỗi phần).

Chỉ trả về định dạng JSON này:
{{
    "overall_score": <số nguyên 0-100>,
    "detailed_scores": {{
        "job_title": {{"score": <số nguyên>, "analysis": "<phân tích ngắn>"}},
        "skills": {{"score": <số nguyên>, "analysis": "<phân tích ngắn>"}},
        "experience": {{"score": <số nguyên>, "analysis": "<phân tích ngắn>"}},
        "education": {{"score": <số nguyên>, "analysis": "<phân tích ngắn>"}}
    }},
    "strengths": ["<điểm mạnh 1>", "<điểm mạnh 2>"],
    "weaknesses": ["<điểm yếu 1>", "<điểm yếu 2>"],
    "summary": "<tóm tắt ngắn gọn>"
}}
"""
        
        try:
            print(f" CV Agent: Gọi Gemini API cho job...")
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
            
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2000,  # Tăng token limit
                    top_p=0.8,
                ),
                safety_settings=safety_settings
            )
            
            # Check if blocked
            if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                if response.prompt_feedback.block_reason:
                    return 0, f"Content blocked by safety filter"
            
            result_text = response.text
            print(f" CV Agent: Raw response length: {len(result_text)}")
            print(f" CV Agent: Raw response preview: {result_text[:200]}...")
            
            # Kiểm tra nếu response bị cắt
            if not result_text.strip().endswith('}') and '{' in result_text:
                print(f" CV Agent: Response appears to be truncated, attempting to fix...")
                # Thêm } cuối nếu thiếu
                if result_text.count('{') > result_text.count('}'):
                    result_text += '}'
            
            # Clean markdown và xử lý JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            # Xử lý JSON với error handling tốt hơn
            try:
                result = json.loads(result_text)
                return result.get("overall_score", 0), result.get("summary", ""), result
            except json.JSONDecodeError as json_err:
                print(f" CV Agent: JSON parsing error: {json_err}")
                print(f" CV Agent: Problematic JSON: {result_text[:500]}...")
                
                # Thử nhiều cách sửa JSON
                json_attempts = [
                    # 1. Loại bỏ ký tự đặc biệt
                    re.sub(r'[^\x20-\x7E]', '', result_text.replace('\n', ' ').replace('\r', ' ')),
                    # 2. Tìm JSON object trong text
                    self._extract_json_from_text(result_text),
                    # 3. Sửa quotes không đúng
                    self._fix_json_quotes(result_text),
                    # 4. Loại bỏ text trước và sau JSON
                    self._extract_clean_json(result_text),
                    # 5. Sửa trailing comma
                    self._fix_trailing_comma(result_text)
                ]
                
                for i, cleaned_json in enumerate(json_attempts):
                    if not cleaned_json:
                        continue
                    try:
                        print(f" CV Agent: Trying JSON fix attempt {i+1}")
                        result = json.loads(cleaned_json)
                        print(f" CV Agent: JSON parsing successful with attempt {i+1}")
                        return result.get("overall_score", 0), result.get("summary", ""), result
                    except Exception as e:
                        print(f" CV Agent: Attempt {i+1} failed: {str(e)[:50]}")
                        continue
                
                # Fallback: tạo kết quả mặc định
                print(f" CV Agent: All JSON parsing attempts failed, using fallback")
                return 0, f"JSON parsing error: {str(json_err)[:100]}", {}
            
        except Exception as e:
            print(f" CV Agent: Lỗi trong compare_cv_job_with_gemini: {e}")
            error_msg = str(e)
            
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                print(f"\n🚨🚨🚨 RATE LIMIT ERROR 429 🚨🚨🚨")
                print(f"📱 Model đang sử dụng: {self.model_name}")
                print(f"❌ Lỗi: {error_msg}")
                print(f"⏰ Thời gian: {datetime.now().strftime('%H:%M:%S')}")
                print(f"🛑 Hệ thống đã dừng phân tích để tránh lỗi API")
                print(f"💡 Giải pháp: Vui lòng thử lại sau 1-2 phút")
                print(f"🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
                return 0, f"🚨 RATE LIMIT ERROR 429: {error_msg[:200]}", {}
            else:
                print(f" Gemini error: {error_msg[:100]}")
                return 0, f"API Error: {error_msg[:100]}", {}
    
    async def process(self, user_input: str, uploaded_files: List[str] = None) -> Dict[str, Any]:
        """
        Xử lý yêu cầu phân tích CV
        """
        try:
            logger.info(f"Xử lý yêu cầu: {user_input}")
            logger.info(f"Uploaded files: {uploaded_files}")
            logger.info(f"Model đang sử dụng: {self.model_name}")
            
            # Nếu có file được upload, so sánh với job requirements
            if uploaded_files and len(uploaded_files) > 0:
                print(" CV Agent: Có file được upload, so sánh với job requirements")
                return await self._compare_uploaded_cv_with_jobs(uploaded_files[0])
            else:
                # Nếu không có file, quét tất cả CV trong thư mục cvs/
                print(" CV Agent: Không có file upload, quét tất cả CV trong thư mục")
                return await self._analyze_all_cvs()
                
        except Exception as e:
            return {
                "agent": "cv_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def _analyze_all_cvs(self) -> Dict[str, Any]:
        """Phân tích tất cả CV trong thư mục cvs/"""
        try:
            print("🔄 CV Agent: Đang quét tất cả CV trong thư mục cvs/...")
            
            # Lấy danh sách tất cả file PDF trong thư mục cvs/
            cv_files = list(self.cv_folder.glob("*.pdf"))
            
            if not cv_files:
                return {
                    "agent": "cv_agent",
                    "status": "success",
                    "result": {
                        "message": "Không tìm thấy CV nào trong thư mục cvs/",
                        "cv_count": 0,
                        "cv_evaluations": []
                    }
                }
            
            print(f" CV Agent: Tìm thấy {len(cv_files)} CV files")
            
            # Đọc job requirements
            job_requirements = self._load_job_requirements()
            
            # Phân tích từng CV
            cv_evaluations = []
            for cv_file in cv_files:
                print(f" CV Agent: Đang phân tích {cv_file.name}")
                evaluation = await self._evaluate_single_cv(str(cv_file), job_requirements)
                cv_evaluations.append(evaluation)
            
            return {
                "agent": "cv_agent",
                "status": "success",
                "result": {
                    "message": f"Đã phân tích {len(cv_files)} CV",
                    "cv_count": len(cv_files),
                    "cv_evaluations": cv_evaluations
                }
            }
            
        except Exception as e:
            return {
                "agent": "cv_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    async def _compare_uploaded_cv_with_jobs(self, uploaded_file: str) -> Dict[str, Any]:
        """So sánh CV được upload với job requirements"""
        try:
            print(f" CV Agent: So sánh CV {uploaded_file} với job requirements")
            
            # Đọc job requirements
            job_requirements = self._load_job_requirements()
            
            # Tạo đường dẫn đầy đủ cho file upload
            if not uploaded_file.startswith('/') and not uploaded_file.startswith('uploads/'):
                cv_path = f"uploads/{uploaded_file}"
            else:
                cv_path = uploaded_file
            
            print(f" CV Agent: Đường dẫn CV: {cv_path}")
            
            # Kiểm tra file có tồn tại không
            if not Path(cv_path).exists():
                return {
                    "agent": "cv_agent",
                    "status": "error",
                    "error": f"File không tồn tại: {cv_path}",
                    "error_type": "FileNotFoundError"
                }
            
            # Phân tích CV được upload
            print(f" CV Agent: Bắt đầu đánh giá CV...")
            evaluation = await self._evaluate_single_cv(cv_path, job_requirements)
            print(f" CV Agent: Hoàn thành đánh giá CV")
            print(f" CV Agent: Evaluation status: {evaluation.get('status')}")
            
            return {
                "agent": "cv_agent",
                "status": "success",
                "result": {
                    "message": f"Đã phân tích CV {uploaded_file}",
                    "cv_count": 1,
                    "cv_evaluations": [evaluation]
                }
            }
            
        except Exception as e:
            return {
                "agent": "cv_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    def _load_job_requirements(self) -> Dict[str, Any]:
        """Đọc job requirements từ Excel file"""
        try:
            if not self.job_file.exists():
                print(f" CV Agent: Không tìm thấy file {self.job_file}")
                return {}
            
            # Đọc Excel file
            df = pd.read_excel(self.job_file)
            print(f" CV Agent: Đã đọc {len(df)} job requirements")
            
            # Convert to dict format
            job_requirements = {}
            for _, row in df.iterrows():
                job_title = row.get('Job Title', '')
                if job_title:
                    job_requirements[job_title] = {
                        'skills_required': row.get('Skills Required', ''),
                        'experience_required': row.get('Experience Required', ''),
                        'education_required': row.get('Education Required', ''),
                        'responsibilities': row.get('Responsibilities', ''),
                        'preferred_keywords': row.get('Preferred Keywords', '')
                    }
            
            return job_requirements
            
        except Exception as e:
            print(f" CV Agent: Lỗi đọc job requirements: {e}")
            return {}

    async def _evaluate_single_cv(self, cv_path: str, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Đánh giá một CV cụ thể"""
        try:
            print(f" CV Agent: Đang đánh giá CV: {cv_path}")
            print(f" CV Agent: Job requirements có {len(job_requirements)} jobs")
            
            # Extract CV content
            cv_data = self.extract_pdf_with_content(cv_path)
            print(f" CV Agent: CV data extracted: {bool(cv_data)}")
            print(f" CV Agent: CV data keys: {list(cv_data.keys()) if cv_data else 'None'}")
            print(f" CV Agent: All text length: {len(cv_data.get('all_text', '')) if cv_data else 0}")
            
            if not cv_data or not cv_data.get('all_text'):
                print(f" CV Agent: Không thể đọc nội dung CV từ {cv_path}")
                print(f" CV Agent: CV data: {cv_data}")
                return {
                    "cv_name": Path(cv_path).name,
                    "status": "error",
                    "error": "Không thể đọc nội dung CV"
                }
            
            cv_text = cv_data['all_text']
            cv_key_info = self.extract_key_info(cv_text)
            
            # So sánh với từng job requirement
            evaluations = []
            for job_title, job_req in job_requirements.items():
                print(f" CV Agent: Đang so sánh với job: {job_title}")
                job_text = f"""
                Job Title: {job_title}
                Skills Required: {job_req.get('skills_required', '')}
                Experience Required: {job_req.get('experience_required', '')}
                Education Required: {job_req.get('education_required', '')}
                Responsibilities: {job_req.get('responsibilities', '')}
                Preferred Keywords: {job_req.get('preferred_keywords', '')}
                """
                
                try:
                    print(f" CV Agent: Bắt đầu đánh giá {job_title}...")
                    # Sử dụng Gemini để đánh giá với phân tích chi tiết
                    score, analysis, detailed_result = self.compare_cv_job_with_gemini(cv_text, job_text, cv_key_info)
                    print(f" CV Agent: Kết quả đánh giá {job_title}: {score}%")
                    
                    # Kiểm tra rate limit
                    if "Rate limit exceeded" in analysis or "429" in analysis:
                        print(f" CV Agent: Rate limit hit! Dừng phân tích...")
                        return {
                            "cv_name": Path(cv_path).name,
                            "status": "error",
                            "error": f"🚨 RATE LIMIT ERROR 429: {analysis}",
                            "cv_key_info": cv_key_info
                        }
                    
                    # Tạo kết quả đánh giá
                    evaluation_result = {
                        "job_title": job_title,
                        "score": score,
                        "analysis": analysis,
                        "detailed_scores": detailed_result.get("detailed_scores", {}),
                        "strengths": detailed_result.get("strengths", []),
                        "weaknesses": detailed_result.get("weaknesses", []),
                        "cv_key_info": cv_key_info
                    }
                    
                    evaluations.append(evaluation_result)
                    
                    # Hiển thị kết quả realtime
                    print(f"\n{'='*60}")
                    print(f"📊 KẾT QUẢ PHÂN TÍCH REALTIME")
                    print(f"{'='*60}")
                    print(f"👤 CV: {Path(cv_path).name}")
                    print(f"💼 Job: {job_title}")
                    print(f"⭐ Điểm số: {score}%")
                    print(f"📝 Phân tích: {analysis}")
                    
                    # Hiển thị điểm chi tiết
                    if detailed_result.get("detailed_scores"):
                        print(f"\n📊 Phân tích chi tiết:")
                        for criteria, data in detailed_result["detailed_scores"].items():
                            criteria_name = {
                                "job_title": "Chức danh",
                                "skills": "Kỹ năng", 
                                "experience": "Kinh nghiệm",
                                "education": "Học vấn"
                            }.get(criteria, criteria)
                            print(f"  - {criteria_name}: {data.get('score', 0)}%")
                    
                    # Hiển thị điểm mạnh/yếu
                    if detailed_result.get("strengths"):
                        print(f"\n✅ Điểm mạnh:")
                        for strength in detailed_result["strengths"][:3]:  # Chỉ hiển thị 3 điểm mạnh đầu
                            print(f"  + {strength}")
                    
                    if detailed_result.get("weaknesses"):
                        print(f"\n❌ Điểm cần cải thiện:")
                        for weakness in detailed_result["weaknesses"][:3]:  # Chỉ hiển thị 3 điểm yếu đầu
                            print(f"  - {weakness}")
                    
                    print(f"{'='*60}\n")
                except Exception as e:
                    print(f" CV Agent: Lỗi đánh giá {job_title}: {e}")
                    print(f" CV Agent: Tiếp tục với job tiếp theo...")
                    evaluations.append({
                        "job_title": job_title,
                        "score": 0,
                        "analysis": f"Lỗi đánh giá: {str(e)}",
                        "detailed_scores": {},
                        "strengths": [],
                        "weaknesses": [],
                        "cv_key_info": cv_key_info
                    })
            
            # Tìm job phù hợp nhất
            best_match = max(evaluations, key=lambda x: x['score'])
            
            # Tính điểm trung bình
            valid_scores = [e['score'] for e in evaluations if isinstance(e['score'], (int, float))]
            average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            
            print(f" CV Agent: Hoàn thành đánh giá {len(evaluations)} jobs")
            print(f" CV Agent: Best match: {best_match['job_title']} ({best_match['score']}%)")
            print(f" CV Agent: Average score: {average_score:.1f}%")
            
            return {
                "cv_name": Path(cv_path).name,
                "cv_key_info": cv_key_info,
                "best_match": best_match,
                "all_evaluations": evaluations,
                "average_score": round(average_score, 2),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "cv_name": Path(cv_path).name,
                "status": "error",
                "error": str(e)
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
    
    def _fix_trailing_comma(self, text: str) -> str:
        """Sửa trailing comma trong JSON"""
        try:
            # Tìm JSON object trong text
            if "```json" in text:
                json_text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_text = text.split("```")[1].split("```")[0].strip()
            else:
                json_text = text.strip()
            
            # Sửa trailing comma
            # Loại bỏ trailing comma trước }
            json_text = re.sub(r',\s*}', '}', json_text)
            # Loại bỏ trailing comma trước ]
            json_text = re.sub(r',\s*]', ']', json_text)
            
            return json_text
        except Exception:
            return ""
    
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
