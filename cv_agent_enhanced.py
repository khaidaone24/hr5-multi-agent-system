import asyncio
import hashlib
import json
import logging
import re
import time
from functools import wraps
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logger.warning("PyMuPDF not installed. PDF processing will be limited.")

class EnhancedCVAgent:
    """Enhanced CV Agent với rate limiting, retry mechanism, và caching"""
    
    def __init__(self):
        self.agent_name = "cv_agent"
        self.model_name = "models/gemini-2.5-flash-lite"
        
        # Rate limiting
        self.request_count = 0
        self.request_window_start = time.time()
        self.max_requests_per_minute = 10
        
        # File size limit (10MB)
        self.max_file_size = 10 * 1024 * 1024
        
        # Cache để tránh xử lý trùng
        self.cv_cache = {}
        
        # Setup LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.1,
            max_output_tokens=2000
        )
        
        # Setup genai
        genai.configure(api_key="AIzaSyBvOkBwv7wjHD5RE1h5abTZgBieUld3o2Y")
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Upload directory
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
        # Job requirements
        self.job_requirements_file = Path("job_requirements/job_requirements.xlsx")
        
        logger.info(f"Enhanced CV Agent initialized with model: {self.model_name}")
    
    def _rate_limit_check(self, func):
        """Decorator for rate limiting"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Reset counter nếu qua window
            if current_time - self.request_window_start > 60:
                self.request_count = 0
                self.request_window_start = current_time
            
            # Check limit
            if self.request_count >= self.max_requests_per_minute:
                wait_time = 60 - (current_time - self.request_window_start)
                logger.warning(f"Rate limit reached. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.request_window_start = time.time()
            
            self.request_count += 1
            return await func(*args, **kwargs)
        
        return wrapper
    
    def _validate_file(self, filepath: Path) -> Tuple[bool, str]:
        """Validate file before processing"""
        if not filepath.exists():
            return False, "File không tồn tại"
        
        if filepath.stat().st_size > self.max_file_size:
            return False, f"File quá lớn (>{self.max_file_size/1024/1024}MB)"
        
        if filepath.suffix.lower() != '.pdf':
            return False, "Chỉ hỗ trợ file PDF"
        
        return True, "OK"
    
    def _get_file_hash(self, filepath: Path) -> str:
        """Get file hash for caching"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Trích xuất text với proper error handling"""
        if not fitz:
            return "Lỗi: PyMuPDF chưa được cài đặt"
        
        doc = None
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num, page in enumerate(doc):
                try:
                    page_text = page.get_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Lỗi đọc trang {page_num + 1}: {e}")
                    continue
            
            return "\n".join(text_parts) if text_parts else "Không thể trích xuất text"
            
        except Exception as e:
            logger.error(f"Lỗi đọc PDF: {str(e)}")
            return f"Lỗi đọc PDF: {str(e)}"
        finally:
            if doc:
                doc.close()
    
    @_rate_limit_check
    async def _extract_cv_info(self, cv_text: str) -> Dict[str, Any]:
        """Extract với rate limiting và retry"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                prompt = f"""
                Phân tích CV và trả về JSON:
                
                {cv_text[:3000]}
                
                Format:
                {{
                    "name": "Tên",
                    "email": "Email",
                    "phone": "SĐT",
                    "skills": ["skill1", "skill2"],
                    "experience_years": "X năm",
                    "education": "Học vấn",
                    "current_position": "Vị trí",
                    "summary": "Tóm tắt"
                }}
                """
                
                response = self.model.generate_content(prompt)
                result_text = response.text
                
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError("Không tìm thấy JSON trong response")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: {e}")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    return self._get_default_cv_info(f"Lỗi sau {max_retries} lần thử: {e}")
        
        return self._get_default_cv_info("Unknown error")
    
    def _get_default_cv_info(self, error_msg: str) -> Dict[str, Any]:
        """Default CV info structure"""
        return {
            "name": "Unknown",
            "email": "Unknown", 
            "phone": "Unknown",
            "skills": [],
            "experience_years": "Unknown",
            "education": "Unknown",
            "current_position": "Unknown",
            "summary": error_msg
        }
    
    @_rate_limit_check
    async def _analyze_cv_with_ai(self, cv_text: str, user_input: str) -> str:
        """AI analysis với rate limiting"""
        try:
            prompt = f"""
            Phân tích CV này và đưa ra đánh giá:
            
            CV: {cv_text[:2000]}
            Yêu cầu: {user_input}
            
            Đưa ra đánh giá ngắn gọn về:
            1. Điểm mạnh
            2. Điểm yếu  
            3. Khuyến nghị
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return f"Lỗi phân tích AI: {str(e)}"
    
    async def _process_uploaded_files(self, uploaded_files: List[str], user_input: str) -> Dict[str, Any]:
        """Process với validation và caching"""
        results = []
        
        for filename in uploaded_files:
            filepath = self.upload_dir / filename
        
            # Validate file
            is_valid, msg = self._validate_file(filepath)
            if not is_valid:
                logger.error(f"File validation failed for {filename}: {msg}")
                results.append({"filename": filename, "error": msg})
                continue
            
            # Check cache
            file_hash = self._get_file_hash(filepath)
            if file_hash in self.cv_cache:
                logger.info(f"Using cache for {filename}")
                results.append(self.cv_cache[file_hash])
                continue
            
            logger.info(f"Processing file {filename}")
            
            # Extract text
            cv_text = self._extract_text_from_pdf(str(filepath))
            if cv_text.startswith("Lỗi"):
                results.append({"filename": filename, "error": cv_text})
                continue
            
            # Extract info với rate limiting
            cv_info = await self._extract_cv_info(cv_text)
            
            # AI analysis với rate limiting
            ai_analysis = await self._analyze_cv_with_ai(cv_text, user_input)
            
            result = {
                "filename": filename,
                "cv_info": cv_info,
                "ai_analysis": ai_analysis,
                "text_length": len(cv_text)
            }
            
            # Cache result
            self.cv_cache[file_hash] = result
            results.append(result)
        
        return {
            "agent": "cv_agent",
            "status": "success",
            "result": {
                "uploaded_files_analysis": results,
                "total_files": len(uploaded_files),
                "timestamp": time.time()
            }
        }
    
    async def process(self, user_input: str, uploaded_files: List[str] = None) -> Dict[str, Any]:
        """Main processing method"""
        logger.info(f"Enhanced CV Agent: Processing request: {user_input}")
        
        if uploaded_files:
            logger.info(f"Processing {len(uploaded_files)} uploaded files")
            return await self._process_uploaded_files(uploaded_files, user_input)
        else:
            return {
                "agent": "cv_agent",
                "status": "error",
                "result": {
                    "error": "No files provided for analysis"
                }
            }