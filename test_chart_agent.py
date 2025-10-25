#!/usr/bin/env python3
"""
Test script cho Chart Agent với dữ liệu thực tế từ Query Agent
"""

import asyncio
import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from chart_agent import ChartAgent

async def test_chart_agent_with_real_data():
    """Test Chart Agent với dữ liệu thực tế từ Query Agent"""
    print("Testing Chart Agent with Real Query Data")
    print("="*60)
    
    chart_agent = ChartAgent()
    
    # Dữ liệu thực tế từ Query Agent (dạng string)
    real_query_data = """Dưới đây là kết quả và biểu đồ so sánh số lượng nhân viên theo phòng ban:**Kết quả:**

| ten_phong_ban | so_luong_nhan_vien |
|---|---|
| Phòng Nhân sự | 9 |
| Phòng IT | 7 |
| Phòng Kế toán | 9 |

**Biểu đồ:**

Do tôi không có khả năng vẽ biểu đồ trực tiếp, bạn có thể tạo biểu đồ cột hoặc biểu đồ tròn bằng cách sử dụng dữ liệu trên trong các công cụ như Excel, Google Sheets, hoặc các thư viện vẽ biểu đồ trong Python (ví dụ: matplotlib, seaborn)."""
    
    test_cases = [
        {
            "input": "Tạo biểu đồ cột so sánh số nhân viên theo phòng ban",
            "data": real_query_data,
            "expected_chart_type": "bar"
        },
        {
            "input": "Vẽ biểu đồ tròn cho thống kê phòng ban",
            "data": real_query_data,
            "expected_chart_type": "pie"
        },
        {
            "input": "Tạo biểu đồ đường cho xu hướng nhân viên",
            "data": real_query_data,
            "expected_chart_type": "line"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['input']}")
        print(f"Expected Chart Type: {test_case['expected_chart_type']}")
        print("-" * 40)
        
        try:
            result = await chart_agent.process(test_case['input'], test_case['data'])
            
            print("✅ Result:")
            print(f"Status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                result_data = result.get('result', {})
                if 'chart_info' in result_data:
                    chart_info = result_data['chart_info']
                    print(f"Chart File: {chart_info.get('chart_file', 'N/A')}")
                    print(f"Chart Type: {chart_info.get('chart_type', 'N/A')}")
                    print(f"Title: {chart_info.get('title', 'N/A')}")
                
                if 'data_analysis' in result_data:
                    analysis = result_data['data_analysis']
                    if analysis and 'basic_info' in analysis:
                        basic_info = analysis['basic_info']
                        print(f"Data Rows: {basic_info.get('total_rows', 'N/A')}")
                        print(f"Data Columns: {basic_info.get('total_columns', 'N/A')}")
                
            elif result.get('status') == 'error':
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("-" * 40)

async def test_chart_agent_data_parsing():
    """Test khả năng parse dữ liệu của Chart Agent"""
    print("\n🧪 Testing Chart Agent Data Parsing")
    print("="*60)
    
    chart_agent = ChartAgent()
    
    # Các loại dữ liệu khác nhau từ Query Agent
    test_data_cases = [
        {
            "name": "Markdown Table Data",
            "data": """| ten_phong_ban | so_luong_nhan_vien |
|---|---|
| Phòng Nhân sự | 9 |
| Phòng IT | 7 |
| Phòng Kế toán | 9 |""",
            "expected_columns": ["ten_phong_ban", "so_luong_nhan_vien"]
        },
        {
            "name": "JSON Data",
            "data": """[{"ten_phong_ban": "Phòng Nhân sự", "so_luong_nhan_vien": 9}, {"ten_phong_ban": "Phòng IT", "so_luong_nhan_vien": 7}]""",
            "expected_columns": ["ten_phong_ban", "so_luong_nhan_vien"]
        },
        {
            "name": "Text Pattern Data",
            "data": """Phòng Nhân sự: 9 nhân viên
Phòng IT: 7 nhân viên
Phòng Kế toán: 9 nhân viên""",
            "expected_columns": ["PhongBan", "SoLuong"]
        },
        {
            "name": "Structured Dict Data",
            "data": {
                "columns": ["PhongBan", "SoLuong"],
                "data": [["Phòng Nhân sự", 9], ["Phòng IT", 7], ["Phòng Kế toán", 9]]
            },
            "expected_columns": ["PhongBan", "SoLuong"]
        }
    ]
    
    for i, test_case in enumerate(test_data_cases, 1):
        print(f"\n🧪 Data Parse Test {i}: {test_case['name']}")
        print(f"Expected Columns: {test_case['expected_columns']}")
        print("-" * 40)
        
        try:
            # Test data normalization
            normalized_data = chart_agent._normalize_data(test_case['data'])
            
            print("📊 Normalized Data:")
            print(f"  Columns: {normalized_data.get('columns', [])}")
            print(f"  Data Rows: {len(normalized_data.get('data', []))}")
            print(f"  Data: {normalized_data.get('data', [])}")
            
            # Check if parsing was successful
            if normalized_data.get('data') and normalized_data.get('columns'):
                print("✅ Data parsing successful!")
                
                # Check if columns match expected
                if normalized_data['columns'] == test_case['expected_columns']:
                    print("✅ Columns match expected!")
                else:
                    print(f"⚠️ Columns don't match expected: {test_case['expected_columns']}")
            else:
                print("❌ Data parsing failed!")
            
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("-" * 40)

async def test_chart_creation():
    """Test tạo biểu đồ với dữ liệu thực tế"""
    print("\n🧪 Testing Chart Creation")
    print("="*60)
    
    chart_agent = ChartAgent()
    
    # Dữ liệu test
    test_data = {
        "columns": ["PhongBan", "SoLuong"],
        "data": [
            ["Phòng Nhân sự", 9],
            ["Phòng IT", 7],
            ["Phòng Kế toán", 9]
        ]
    }
    
    chart_types = ["bar", "pie", "line"]
    
    for chart_type in chart_types:
        print(f"\n🧪 Creating {chart_type} chart")
        print("-" * 30)
        
        try:
            result = chart_agent._create_chart(
                test_data,
                chart_type=chart_type,
                title=f"Biểu đồ {chart_type} - Số nhân viên theo phòng ban"
            )
            
            if result.get('success'):
                print(f"✅ {chart_type.title()} chart created successfully!")
                print(f"  Chart file: {result.get('chart_file', 'N/A')}")
                print(f"  Title: {result.get('title', 'N/A')}")
            else:
                print(f"❌ Failed to create {chart_type} chart: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ Exception creating {chart_type} chart: {e}")
        
        print("-" * 30)

async def main():
    """Main test function"""
    print("Testing Enhanced Chart Agent")
    print("="*60)
    
    tests = [
        ("Chart Agent with Real Data", test_chart_agent_with_real_data),
        ("Data Parsing", test_chart_agent_data_parsing),
        ("Chart Creation", test_chart_creation)
    ]
    
    print("\nAvailable tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"{i}. {name}")
    
    print("\n0. Run all tests")
    
    choice = input("\nSelect test (0-3): ").strip()
    
    if choice == "0":
        # Run all tests
        for name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"🎬 Running Test: {name}")
            print(f"{'='*60}")
            try:
                await test_func()
            except Exception as e:
                print(f"❌ Test '{name}' failed: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(tests):
        # Run specific test
        test_name, test_func = tests[int(choice) - 1]
        print(f"\n🎬 Running Test: {test_name}")
        try:
            await test_func()
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print("❌ Invalid choice!")
    
    print("\n🎉 Chart Agent testing completed!")

if __name__ == "__main__":
    asyncio.run(main())
