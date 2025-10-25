#!/usr/bin/env python3
"""
Demo script cho Multi-Agent HR System
"""

import asyncio
import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from main_agent_system import MultiAgentSystem

async def demo_basic_workflow():
    """Demo workflow cơ bản với orchestrator mới"""
    print("🎬 Demo: Basic Workflow with LLM-based Orchestrator")
    print("="*60)
    
    system = MultiAgentSystem()
    
    # Test cases
    test_cases = [
        {
            "input": "Tìm nhân viên có lương cao nhất",
            "expected_agents": ["query_agent"],
            "description": "Query database để tìm nhân viên lương cao"
        },
        {
            "input": "Phân tích CV của ứng viên Python developer", 
            "expected_agents": ["cv_agent"],
            "description": "Phân tích CV ứng viên"
        },
        {
            "input": "Tạo biểu đồ thống kê nhân viên theo phòng ban",
            "expected_agents": ["query_agent", "chart_agent"], 
            "description": "Tạo biểu đồ trực quan hóa dữ liệu (cần query trước)"
        },
        {
            "input": "Tổng hợp báo cáo về tình hình nhân sự",
            "expected_agents": ["query_agent", "analysis_agent"],
            "description": "Phân tích tổng hợp"
        },
        {
            "input": "Tạo biểu đồ mà không có dữ liệu",
            "expected_agents": ["query_agent", "chart_agent"],
            "description": "Test case đặc biệt: chart cần data từ query"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['description']}")
        print(f"Input: {test_case['input']}")
        print(f"Expected Agents: {test_case['expected_agents']}")
        
        try:
            result = await system.process_single_request(test_case['input'])
            
            print("✅ Result:")
            print(f"Final Status: {result.get('final_status', 'unknown')}")
            print(f"Success Rate: {result.get('success_rate', 0):.2%}")
            
            # Show execution summary
            if 'execution_summary' in result:
                summary = result['execution_summary']
                print(f"Execution: {summary['successful_steps']}/{summary['total_steps']} steps successful")
            
            # Show agent results
            if 'agent_results' in result:
                print("📋 Agent Execution:")
                for agent_result in result['agent_results']:
                    print(f"  Step {agent_result['step']}: {agent_result['agent']} - {agent_result['status']}")
                    if agent_result['status'] == 'error':
                        print(f"    Error: {agent_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 60)

async def demo_workflow_types():
    """Demo các loại workflow khác nhau"""
    print("\n🎬 Demo: Different Workflow Types")
    print("="*50)
    
    system = MultiAgentSystem()
    
    workflows = [
        {
            "name": "Query then Chart",
            "type": "query_then_chart",
            "input": "Truy vấn dữ liệu nhân viên và tạo biểu đồ",
            "description": "Query dữ liệu rồi tạo biểu đồ"
        },
        {
            "name": "CV Analysis", 
            "type": "cv_analysis",
            "input": "Phân tích CV và tìm ứng viên phù hợp",
            "description": "Phân tích CV và ứng viên"
        },
        {
            "name": "Full Analysis",
            "type": "full_analysis", 
            "input": "Phân tích toàn diện về nhân sự",
            "description": "Phân tích với tất cả agents"
        }
    ]
    
    for workflow in workflows:
        print(f"\n🔄 Workflow: {workflow['name']}")
        print(f"Description: {workflow['description']}")
        print(f"Input: {workflow['input']}")
        
        try:
            result = await system.process_workflow(workflow['input'], workflow['type'])
            
            print("✅ Result:")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Workflow: {result.get('workflow', 'unknown')}")
            
            # Show results from each agent
            for key, value in result.items():
                if key.endswith('_result') and isinstance(value, dict):
                    agent_name = key.replace('_result', '')
                    print(f"  {agent_name}: {value.get('status', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)

async def demo_individual_agents():
    """Demo từng agent riêng lẻ"""
    print("\n🎬 Demo: Individual Agents")
    print("="*50)
    
    system = MultiAgentSystem()
    
    # Test data for chart agent
    test_data = {
        "columns": ["PhongBan", "SoLuong", "LuongTB"],
        "data": [
            ["IT", 15, 12000000],
            ["HR", 8, 8000000], 
            ["Finance", 12, 10000000],
            ["Marketing", 10, 9000000]
        ]
    }
    
    # Test cases for each agent
    agent_tests = [
        {
            "name": "Query Agent",
            "agent": system.query_agent,
            "input": "Liệt kê tất cả nhân viên",
            "description": "Test truy vấn database"
        },
        {
            "name": "CV Agent",
            "agent": system.cv_agent,
            "input": "Phân tích tất cả CV trong thư mục",
            "description": "Test phân tích CV"
        },
        {
            "name": "Chart Agent", 
            "agent": system.chart_agent,
            "input": "Tạo biểu đồ cột cho dữ liệu phòng ban",
            "data": test_data,
            "description": "Test tạo biểu đồ"
        },
        {
            "name": "Analysis Agent",
            "agent": system.analysis_agent,
            "input": "Phân tích kết quả từ query và CV agent",
            "data": [
                {"agent": "query_agent", "status": "success", "result": {"data": "test"}},
                {"agent": "cv_agent", "status": "success", "result": {"total_cvs": 5}}
            ],
            "description": "Test phân tích tổng hợp"
        }
    ]
    
    for test in agent_tests:
        print(f"\n🤖 Testing {test['name']}")
        print(f"Description: {test['description']}")
        print(f"Input: {test['input']}")
        
        try:
            if 'data' in test:
                result = await test['agent'].process(test['input'], test['data'])
            else:
                result = await test['agent'].process(test['input'])
            
            print("✅ Result:")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Agent: {result.get('agent', 'unknown')}")
            
            # Show key information from result
            if 'result' in result:
                result_data = result['result']
                if isinstance(result_data, dict):
                    for key, value in result_data.items():
                        if key in ['total_cvs', 'total_charts', 'chart_file', 'summary']:
                            print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)

async def demo_error_handling():
    """Demo xử lý lỗi"""
    print("\n🎬 Demo: Error Handling")
    print("="*50)
    
    system = MultiAgentSystem()
    
    # Test cases that might cause errors
    error_test_cases = [
        {
            "input": "",
            "description": "Empty input"
        },
        {
            "input": "Invalid request that doesn't match any pattern",
            "description": "Unrecognized input"
        },
        {
            "input": "Truy vấn dữ liệu không tồn tại",
            "description": "Query non-existent data"
        }
    ]
    
    for i, test_case in enumerate(error_test_cases, 1):
        print(f"\n🧪 Error Test {i}: {test_case['description']}")
        print(f"Input: '{test_case['input']}'")
        
        try:
            result = await system.process_single_request(test_case['input'])
            
            print("✅ Result:")
            print(f"Status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'error':
                print(f"Error: {result.get('error', 'Unknown error')}")
            else:
                print("Unexpected: No error occurred")
            
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("-" * 50)

async def demo_performance():
    """Demo performance testing"""
    print("\n🎬 Demo: Performance Testing")
    print("="*50)
    
    system = MultiAgentSystem()
    
    # Test cases for performance
    performance_tests = [
        {
            "input": "Tìm nhân viên có lương cao nhất",
            "expected_time": 10  # seconds
        },
        {
            "input": "Phân tích CV của ứng viên",
            "expected_time": 30  # seconds
        },
        {
            "input": "Tạo biểu đồ thống kê",
            "expected_time": 15  # seconds
        }
    ]
    
    for i, test in enumerate(performance_tests, 1):
        print(f"\n⏱️ Performance Test {i}")
        print(f"Input: {test['input']}")
        print(f"Expected time: {test['expected_time']}s")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await system.process_single_request(test['input'])
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            print(f"✅ Completed in {duration:.2f}s")
            print(f"Status: {result.get('status', 'unknown')}")
            
            if duration > test['expected_time']:
                print(f"⚠️ Warning: Took longer than expected ({duration:.2f}s > {test['expected_time']}s)")
            
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            print(f"❌ Failed after {duration:.2f}s: {e}")
        
        print("-" * 50)

async def main():
    """Main demo function"""
    print("🚀 Multi-Agent HR System Demo")
    print("="*60)
    
    demos = [
        ("Basic Workflow", demo_basic_workflow),
        ("Workflow Types", demo_workflow_types),
        ("Individual Agents", demo_individual_agents),
        ("Error Handling", demo_error_handling),
        ("Performance", demo_performance)
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    
    print("\n0. Run all demos")
    
    choice = input("\nSelect demo (0-5): ").strip()
    
    if choice == "0":
        # Run all demos
        for name, demo_func in demos:
            print(f"\n{'='*60}")
            print(f"🎬 Running Demo: {name}")
            print(f"{'='*60}")
            try:
                await demo_func()
            except Exception as e:
                print(f"❌ Demo '{name}' failed: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(demos):
        # Run specific demo
        demo_name, demo_func = demos[int(choice) - 1]
        print(f"\n🎬 Running Demo: {demo_name}")
        try:
            await demo_func()
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    else:
        print("❌ Invalid choice!")
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
