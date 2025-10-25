#!/usr/bin/env python3
"""
Simple test for Chart Agent
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from chart_agent import ChartAgent

async def test_simple_chart():
    """Simple test for Chart Agent"""
    print("Testing Chart Agent with Simple Data")
    print("="*50)
    
    chart_agent = ChartAgent()
    
    # Test data from Query Agent
    query_data = """| ten_phong_ban | so_luong_nhan_vien |
|---|---|
| Phong Nhan su | 9 |
| Phong IT | 7 |
| Phong Ke toan | 9 |"""
    
    print("Input data:")
    print(query_data)
    print("-" * 30)
    
    try:
        result = await chart_agent.process("Tao bieu do cot", query_data)
        
        print("Result:")
        print(f"Status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            result_data = result.get('result', {})
            if 'chart_info' in result_data:
                chart_info = result_data['chart_info']
                print(f"Chart File: {chart_info.get('chart_file', 'N/A')}")
                print(f"Chart Type: {chart_info.get('chart_type', 'N/A')}")
                print("SUCCESS: Chart created!")
            else:
                print("Result data:", result_data)
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_chart())



