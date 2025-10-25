import asyncio
import json
from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

class ConversationalAgent:
    """
    Conversational Agent - Xử lý các cuộc trò chuyện chung khi không có intent cụ thể
    """
    
    def __init__(self):
        load_dotenv()
        self.GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.llm_model = "models/gemini-2.5-flash-lite"
        self.llm_temperature = 0.7
        
        # Lịch sử conversation
        self.conversation_history = []
        
        print("Conversational Agent initialized")
    
    async def _call_gemini(self, prompt: str) -> str:
        """
        Gọi Gemini API để xử lý conversation
        """
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                google_api_key=self.GEMINI_API_KEY,
                temperature=self.llm_temperature,
            )
            
            response = await llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            print(f" Conversational Agent: Lỗi gọi Gemini: {e}")
            return "Xin lỗi, tôi gặp sự cố kỹ thuật. Bạn có thể thử lại sau không?"
    
    async def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Xử lý cuộc trò chuyện với người dùng
        """
        try:
            print(f" Conversational Agent: Xử lý cuộc trò chuyện: '{user_input}'")
            
            # Tạo context từ lịch sử conversation
            context_str = ""
            if self.conversation_history:
                recent_history = self.conversation_history[-5:]  # Lấy 5 cuộc trò chuyện gần nhất
                context_str = "\n".join([
                    f"Người dùng: {entry['user_input']}\nAI: {entry['response']}\n"
                    for entry in recent_history
                ])
            
            # Tạo prompt cho conversation
            prompt = f"""
Bạn là một AI Assistant thân thiện và hữu ích trong hệ thống HR Multi-Agent. 
Bạn có thể trò chuyện về các chủ đề liên quan đến nhân sự, công việc, hoặc bất kỳ chủ đề nào khác.

Lịch sử cuộc trò chuyện gần đây:
{context_str}

Yêu cầu hiện tại của người dùng: "{user_input}"

Hãy trả lời một cách thân thiện, hữu ích và tự nhiên. Nếu người dùng hỏi về các chức năng của hệ thống HR, hãy giải thích ngắn gọn về:
- Phân tích CV và đánh giá ứng viên
- Truy vấn dữ liệu nhân sự
- Tạo biểu đồ và báo cáo thống kê
- Tổng hợp và phân tích dữ liệu

Trả lời bằng tiếng Việt, ngắn gọn và dễ hiểu.
"""
            
            # Gọi Gemini để tạo response
            response = await self._call_gemini(prompt)
            
            # Lưu vào lịch sử
            self.conversation_history.append({
                "user_input": user_input,
                "response": response,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Giới hạn lịch sử để tránh quá tải
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return {
                "agent": "conversational_agent",
                "status": "success",
                "result": {
                    "response": response,
                    "conversation_id": len(self.conversation_history),
                    "context": context
                }
            }
            
        except Exception as e:
            print(f" Conversational Agent: Lỗi xử lý: {e}")
            return {
                "agent": "conversational_agent",
                "status": "error",
                "error": str(e),
                "result": {
                    "response": "Xin lỗi, tôi gặp sự cố kỹ thuật. Bạn có thể thử lại sau không?",
                    "conversation_id": 0,
                    "context": context
                }
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử cuộc trò chuyện
        """
        return self.conversation_history
    
    def clear_history(self):
        """
        Xóa lịch sử cuộc trò chuyện
        """
        self.conversation_history = []
        print("Conversational Agent: Đã xóa lịch sử cuộc trò chuyện")

# Test function
async def test_conversational_agent():
    agent = ConversationalAgent()
    
    test_cases = [
        "Xin chào!",
        "Hệ thống này có thể làm gì?",
        "Tôi muốn tìm hiểu về phân tích CV",
        "Cảm ơn bạn!",
        "Tạo biểu đồ thống kê nhân viên"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_input}")
        print(f"{'='*60}")
        
        try:
            result = await agent.process(test_input)
            print(f"Status: {result['status']}")
            print(f"Response: {result['result']['response']}")
            
        except Exception as e:
            print(f"Test failed: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(test_conversational_agent())
