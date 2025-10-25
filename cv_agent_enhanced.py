#!/usr/bin/env python3
"""
Enhanced CV Agent với khả năng xử lý file PDF upload
"""

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

class EnhancedCVAgent:
    """
    Enhanced CV Agent - Xử lý CV với khả năng upload file PDF
    """
    
    def __init__(self):
        load_dotenv()
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.gemini_api_key:
            raise ValueError("⚠️ Thiếu GOOGLE_API_KEY trong .env")
        
        # Khởi tạo Gemini AI
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Khởi tạo LangChain LLM
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash-lite",
            google_api_key=self.gemini_api_key,
            temperature=0.1,
        )
        
        # Thư mục lưu CVs
        self.cv_dir = Path("cvs")
        self.cv_dir.mkdir(exist_ok=True)
        
        # Thư mục uploads
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    def _has_cv_intent(self, user_input: str) -> bool:
        """Kiểm tra xem yêu cầu có liên quan đến CV không"""
        cv_keywords = [
            'cv', 'resume', 'hồ sơ', 'ứng viên', 'candidate', 
            'phân tích cv', 'đánh giá cv', 'so sánh cv',
            'tuyển dụng', 'recruitment', 'job application',
            'phỏng vấn', 'interview', 'screening'
        ]
        user_input_lower = user_input.lower()
        return any(keyword in user_input_lower for keyword in cv_keywords)
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Trích xuất text từ file PDF"""
        if not fitz:
            return "Lỗi: PyMuPDF chưa được cài đặt"
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            return f"Lỗi đọc PDF: {str(e)}"
    
    def _extract_cv_info(self, cv_text: str) -> Dict[str, Any]:
        """Trích xuất thông tin quan trọng từ CV"""
        try:
            prompt = f"""
            Phân tích CV sau và trích xuất thông tin quan trọng:
            
            {cv_text[:3000]}  # Giới hạn độ dài để tránh lỗi
            
            Hãy trả về JSON với format:
            {{
                "name": "Tên ứng viên",
                "email": "Email",
                "phone": "Số điện thoại",
                "skills": ["skill1", "skill2", ...],
                "experience_years": "Số năm kinh nghiệm",
                "education": "Học vấn",
                "current_position": "Vị trí hiện tại",
                "summary": "Tóm tắt ngắn gọn"
            }}
            """
            
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # Tìm JSON trong response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "name": "Unknown",
                    "email": "Unknown",
                    "phone": "Unknown",
                    "skills": [],
                    "experience_years": "Unknown",
                    "education": "Unknown",
                    "current_position": "Unknown",
                    "summary": "Không thể phân tích CV"
                }
                
        except Exception as e:
            return {
                "name": "Unknown",
                "email": "Unknown", 
                "phone": "Unknown",
                "skills": [],
                "experience_years": "Unknown",
                "education": "Unknown",
                "current_position": "Unknown",
                "summary": f"Lỗi phân tích: {str(e)}"
            }
    
    async def _analyze_cv_with_ai(self, cv_text: str, user_requirement: str) -> str:
        """Phân tích CV bằng AI theo yêu cầu cụ thể"""
        try:
            prompt = f"""
            Bạn là một chuyên gia HR với 10 năm kinh nghiệm.
            
            Yêu cầu: {user_requirement}
            
            CV cần phân tích:
            {cv_text[:2000]}
            
            Hãy phân tích CV này theo yêu cầu và đưa ra:
            1. Đánh giá tổng quan
            2. Điểm mạnh
            3. Điểm yếu
            4. Khuyến nghị
            5. Điểm phù hợp (1-10)
            
            Trả về kết quả có cấu trúc và dễ hiểu.
            """
            
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return f"Lỗi phân tích AI: {str(e)}"
    
    async def _process_uploaded_files(self, uploaded_files: List[str], user_input: str) -> Dict[str, Any]:
        """Xử lý các file PDF đã upload"""
        try:
            results = []
            
            for filename in uploaded_files:
                filepath = self.upload_dir / filename
                if filepath.exists():
                    print(f"📄 CV Agent: Phân tích file {filename}")
                    
                    # Trích xuất text từ PDF
                    cv_text = self._extract_text_from_pdf(str(filepath))
                    
                    if cv_text.startswith("Lỗi"):
                        results.append({
                            "filename": filename,
                            "error": cv_text
                        })
                        continue
                    
                    # Trích xuất thông tin cơ bản
                    cv_info = self._extract_cv_info(cv_text)
                    
                    # Phân tích bằng AI
                    ai_analysis = await self._analyze_cv_with_ai(cv_text, user_input)
                    
                    results.append({
                        "filename": filename,
                        "cv_info": cv_info,
                        "ai_analysis": ai_analysis,
                        "text_length": len(cv_text)
                    })
                else:
                    results.append({
                        "filename": filename,
                        "error": "File không tồn tại"
                    })
            
            return {
                "agent": "cv_agent",
                "status": "success",
                "result": {
                    "uploaded_files_analysis": results,
                    "total_files": len(uploaded_files),
                    "timestamp": asyncio.get_event_loop().time()
                }
            }
            
        except Exception as e:
            return {
                "agent": "cv_agent",
                "status": "error",
                "error": f"Lỗi xử lý file upload: {str(e)}"
            }
    
    async def _scan_all_cvs(self, user_input: str) -> Dict[str, Any]:
        """Quét tất cả CV có sẵn trong hệ thống"""
        try:
            # Tìm tất cả file PDF trong thư mục CVs
            pdf_files = list(self.cv_dir.glob("*.pdf"))
            
            if not pdf_files:
                return {
                    "agent": "cv_agent",
                    "status": "info",
                    "result": {
                        "message": "Không tìm thấy file CV nào trong hệ thống",
                        "suggestion": "Hãy upload file PDF hoặc đặt CV vào thư mục 'cvs/'"
                    }
                }
            
            print(f"📁 CV Agent: Tìm thấy {len(pdf_files)} file CV, bắt đầu phân tích...")
            
            results = []
            for pdf_file in pdf_files:
                print(f"📄 CV Agent: Phân tích {pdf_file.name}")
                
                # Trích xuất text từ PDF
                cv_text = self._extract_text_from_pdf(str(pdf_file))
                
                if cv_text.startswith("Lỗi"):
                    results.append({
                        "filename": pdf_file.name,
                        "error": cv_text
                    })
                    continue
                
                # Trích xuất thông tin cơ bản
                cv_info = self._extract_cv_info(cv_text)
                
                # Phân tích bằng AI
                ai_analysis = await self._analyze_cv_with_ai(cv_text, user_input)
                
                results.append({
                    "filename": pdf_file.name,
                    "cv_info": cv_info,
                    "ai_analysis": ai_analysis,
                    "text_length": len(cv_text)
                })
            
            return {
                "agent": "cv_agent",
                "status": "success",
                "result": {
                    "all_cvs_analysis": results,
                    "total_cvs": len(pdf_files),
                    "timestamp": asyncio.get_event_loop().time()
                }
            }
            
        except Exception as e:
            return {
                "agent": "cv_agent",
                "status": "error",
                "error": f"Lỗi quét tất cả CV: {str(e)}"
            }
    
    async def process(self, user_input: str, uploaded_files: List[str] = None) -> Dict[str, Any]:
        """
        Xử lý yêu cầu phân tích CV với khả năng xử lý file upload
        """
        try:
            print(f"🔍 CV Agent: Xử lý yêu cầu '{user_input}'")
            
            # Xử lý file upload nếu có
            if uploaded_files:
                print(f"📁 CV Agent: Xử lý {len(uploaded_files)} file(s) đã upload")
                return await self._process_uploaded_files(uploaded_files, user_input)
            else:
                # Nếu không có file upload nhưng intent có CV, quét tất cả CV
                if self._has_cv_intent(user_input):
                    print("📁 CV Agent: Không có file upload, quét tất cả CV có sẵn")
                    return await self._scan_all_cvs(user_input)
                else:
                    return {
                        "agent": "cv_agent",
                        "status": "info",
                        "result": {
                            "message": "CV Agent sẵn sàng phân tích CV",
                            "usage": "Upload file PDF hoặc yêu cầu phân tích CV cụ thể",
                            "capabilities": [
                                "Phân tích CV từ file PDF",
                                "Trích xuất thông tin quan trọng",
                                "So sánh với yêu cầu công việc",
                                "Đánh giá và khuyến nghị"
                            ]
                        }
                    }
            
        except Exception as e:
            return {
                "agent": "cv_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

# Test function
async def test_enhanced_cv_agent():
    """Test Enhanced CV Agent"""
    print("Testing Enhanced CV Agent")
    print("="*50)
    
    agent = EnhancedCVAgent()
    
    # Test 1: Không có file upload, không có CV intent
    print("\nTest 1: Không có CV intent")
    result1 = await agent.process("Tìm nhân viên có lương cao nhất")
    print(f"Result: {result1['status']}")
    
    # Test 2: Có CV intent nhưng không có file upload
    print("\nTest 2: Có CV intent, không có file upload")
    result2 = await agent.process("Phân tích CV của ứng viên Python developer")
    print(f"Result: {result2['status']}")
    
    # Test 3: Có file upload
    print("\nTest 3: Có file upload")
    result3 = await agent.process("Phân tích CV này", ["test_cv.pdf"])
    print(f"Result: {result3['status']}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_cv_agent())
