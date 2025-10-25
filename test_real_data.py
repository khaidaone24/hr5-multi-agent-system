#!/usr/bin/env python3
"""
Test Chart Agent với dữ liệu thực tế từ Query Agent
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from chart_agent import ChartAgent

async def test_with_real_query_data():
    """Test với dữ liệu thực tế từ Query Agent"""
    print("Testing Chart Agent with Real Query Data")
    print("="*60)
    
    chart_agent = ChartAgent()
    
    # Dữ liệu thực tế từ Query Agent (có thêm text mô tả)
    real_data = """Dưới đây là kết quả phân tích lương trung bình theo phòng ban:

| ten_phong_ban | luong_trung_binh |
|---|---|
| Phòng Nhân sự | 12888888.89 |
| Phòng IT | 10857142.86 |
| Phòng Kế toán | 12222222.22 |

Do công cụ hiện tại không có khả năng vẽ biểu đồ, tôi không thể trực tiếp tạo biểu đồ từ kết quả này. Tuy nhiên, bạn có thể dễ dàng tạo biểu đồ bằng cách sử dụng các công cụ trực quan hóa dữ liệu khác như Excel, Google Sheets, hoặc các thư viện Python như Matplotlib hoặc Seaborn. Bạn chỉ cần nhập dữ liệu trên vào các công cụ này để tạo biểu đồ mong muốn."""
    
    print("Input data:")
    print("Real query data with table...")
    print("-" * 40)
    
    try:
        result = await chart_agent.process("Tao bieu do cot cho luong trung binh", real_data)
        
        print("Result:")
        print(f"Status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            result_data = result.get('result', {})
            if 'chart_info' in result_data:
                chart_info = result_data['chart_info']
                print(f"Chart File: {chart_info.get('chart_file', 'N/A')}")
                print(f"Chart Type: {chart_info.get('chart_type', 'N/A')}")
                print("SUCCESS: Chart created with real data!")
            else:
                print("Result data:", result_data)
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_with_real_query_data())
