#!/usr/bin/env python3
"""
Test script cho Orchestrator Agent mới với LLM-based intent analysis
"""

import asyncio
import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from orchestrator_agent import OrchestratorAgent

async def test_llm_intent_analysis():
    """Test LLM-based intent analysis"""
    print("🧪 Testing LLM-based Intent Analysis")
    print("="*60)
    
    orchestrator = OrchestratorAgent()
    
    test_cases = [
        {
            "input": "Tìm nhân viên có lương cao nhất",
            "expected_intent": "query",
            "expected_agents": ["query_agent"]
        },
        {
            "input": "Phân tích CV của ứng viên Python developer",
            "expected_intent": "cv_analysis", 
            "expected_agents": ["cv_agent"]
        },
        {
            "input": "Tạo biểu đồ thống kê nhân viên theo phòng ban",
            "expected_intent": "chart_creation",
            "expected_agents": ["query_agent", "chart_agent"]
        },
        {
            "input": "Tạo biểu đồ mà không có dữ liệu",
            "expected_intent": "chart_creation_without_data",
            "expected_agents": ["query_agent", "chart_agent"]
        },
        {
            "input": "Tổng hợp báo cáo về tình hình nhân sự",
            "expected_intent": "comprehensive_analysis",
            "expected_agents": ["query_agent", "analysis_agent"]
        },
        {
            "input": "So sánh CV với yêu cầu công việc Data Analyst",
            "expected_intent": "cv_job_matching",
            "expected_agents": ["cv_agent"]
        },
        {
            "input": "Tạo biểu đồ cột cho dữ liệu lương nhân viên",
            "expected_intent": "chart_creation",
            "expected_agents": ["query_agent", "chart_agent"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['input']}")
        print(f"Expected Intent: {test_case['expected_intent']}")
        print(f"Expected Agents: {test_case['expected_agents']}")
        print("-" * 40)
        
        try:
            # Test intent analysis only
            intent_analysis = await orchestrator.analyze_intent(test_case['input'])
            
            print("📊 Intent Analysis Result:")
            print(f"  Primary Intent: {intent_analysis.get('primary_intent', 'unknown')}")
            print(f"  Required Agents: {intent_analysis.get('required_agents', [])}")
            print(f"  Confidence: {intent_analysis.get('confidence', 0):.2f}")
            print(f"  Reasoning: {intent_analysis.get('reasoning', 'N/A')}")
            
            # Check execution plan
            execution_plan = intent_analysis.get('execution_plan', [])
            print(f"  Execution Plan:")
            for step in execution_plan:
                print(f"    Step {step.get('step', 0)}: {step.get('agent', 'unknown')} - {step.get('reason', 'N/A')}")
            
            # Check special requirements
            special_reqs = intent_analysis.get('special_requirements', {})
            print(f"  Special Requirements:")
            print(f"    Needs Data: {special_reqs.get('needs_data', False)}")
            print(f"    Needs Chart: {special_reqs.get('needs_chart', False)}")
            print(f"    Needs Analysis: {special_reqs.get('needs_analysis', False)}")
            
        except Exception as e:
            print(f"❌ Error in intent analysis: {e}")
        
        print("-" * 40)

async def test_multi_agent_execution():
    """Test multi-agent execution"""
    print("\n🧪 Testing Multi-Agent Execution")
    print("="*60)
    
    orchestrator = OrchestratorAgent()
    
    test_cases = [
        {
            "input": "Tìm nhân viên có lương cao nhất",
            "description": "Simple query - should call only query_agent"
        },
        {
            "input": "Tạo biểu đồ thống kê nhân viên theo phòng ban",
            "description": "Chart creation - should call query_agent first, then chart_agent"
        },
        {
            "input": "Tạo biểu đồ mà không có dữ liệu",
            "description": "Special case - should call query_agent first to get data"
        },
        {
            "input": "Phân tích CV và tạo báo cáo tổng hợp",
            "description": "Complex workflow - should call multiple agents"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['description']}")
        print(f"Input: {test_case['input']}")
        print("-" * 40)
        
        try:
            # Full execution
            result = await orchestrator.process(test_case['input'])
            
            print("✅ Execution Result:")
            print(f"  Final Status: {result.get('final_status', 'unknown')}")
            print(f"  Success Rate: {result.get('success_rate', 0):.2%}")
            print(f"  Total Agents: {result.get('total_agents_executed', 0)}")
            
            # Show execution summary
            if 'execution_summary' in result:
                summary = result['execution_summary']
                print(f"  Execution Summary:")
                print(f"    Total Steps: {summary['total_steps']}")
                print(f"    Successful: {summary['successful_steps']}")
                print(f"    Failed: {summary['failed_steps']}")
            
            # Show agent results
            if 'agent_results' in result:
                print(f"  Agent Results:")
                for agent_result in result['agent_results']:
                    status_icon = "✅" if agent_result['status'] == 'success' else "❌"
                    print(f"    {status_icon} Step {agent_result['step']}: {agent_result['agent']} - {agent_result['status']}")
                    if agent_result['status'] == 'error':
                        print(f"      Error: {agent_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ Error in execution: {e}")
        
        print("-" * 40)

async def test_special_chart_case():
    """Test đặc biệt cho trường hợp chart cần data"""
    print("\n🧪 Testing Special Chart Case (No Data)")
    print("="*60)
    
    orchestrator = OrchestratorAgent()
    
    # Test cases đặc biệt cho chart
    chart_test_cases = [
        "Tạo biểu đồ thống kê nhân viên",
        "Vẽ biểu đồ cột cho dữ liệu lương",
        "Tạo chart phân tích phòng ban",
        "Biểu đồ tròn cho thống kê chức vụ",
        "Tạo visualization cho dữ liệu HR"
    ]
    
    for i, test_input in enumerate(chart_test_cases, 1):
        print(f"\n🧪 Chart Test {i}: {test_input}")
        print("-" * 40)
        
        try:
            # Test intent analysis
            intent_analysis = await orchestrator.analyze_intent(test_input)
            
            print("📊 Intent Analysis:")
            print(f"  Primary Intent: {intent_analysis.get('primary_intent', 'unknown')}")
            print(f"  Required Agents: {intent_analysis.get('required_agents', [])}")
            
            # Check if query_agent is called first
            execution_plan = intent_analysis.get('execution_plan', [])
            if execution_plan:
                first_agent = execution_plan[0].get('agent', '')
                print(f"  First Agent: {first_agent}")
                
                if first_agent == 'query_agent':
                    print("  ✅ Correct: Query agent called first to get data")
                else:
                    print("  ⚠️ Warning: Query agent not called first")
            
            # Check special requirements
            special_reqs = intent_analysis.get('special_requirements', {})
            if special_reqs.get('needs_data', False):
                print("  ✅ Correct: System detected need for data")
            else:
                print("  ⚠️ Warning: System didn't detect need for data")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)

async def test_error_handling():
    """Test error handling"""
    print("\n🧪 Testing Error Handling")
    print("="*60)
    
    orchestrator = OrchestratorAgent()
    
    error_test_cases = [
        "",
        "Invalid request that doesn't make sense",
        "Truy vấn dữ liệu không tồn tại",
        "Request with special characters: @#$%^&*()",
        "Very long request " + "x" * 1000
    ]
    
    for i, test_input in enumerate(error_test_cases, 1):
        print(f"\n🧪 Error Test {i}: '{test_input[:50]}{'...' if len(test_input) > 50 else ''}'")
        print("-" * 40)
        
        try:
            result = await orchestrator.process(test_input)
            
            print("✅ Result:")
            print(f"  Final Status: {result.get('final_status', 'unknown')}")
            print(f"  Success Rate: {result.get('success_rate', 0):.2%}")
            
            # Check if system handled error gracefully
            if result.get('final_status') in ['success', 'partial_success']:
                print("  ✅ System handled gracefully")
            else:
                print("  ⚠️ System may have issues")
            
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("-" * 40)

async def main():
    """Main test function"""
    print("🚀 Testing Enhanced Orchestrator Agent")
    print("="*60)
    
    tests = [
        ("LLM Intent Analysis", test_llm_intent_analysis),
        ("Multi-Agent Execution", test_multi_agent_execution),
        ("Special Chart Case", test_special_chart_case),
        ("Error Handling", test_error_handling)
    ]
    
    print("\nAvailable tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"{i}. {name}")
    
    print("\n0. Run all tests")
    
    choice = input("\nSelect test (0-4): ").strip()
    
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
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    asyncio.run(main())
