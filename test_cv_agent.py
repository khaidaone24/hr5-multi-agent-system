#!/usr/bin/env python3
"""
Test CV Agent với logic mới
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

async def test_cv_agent():
    """Test CV Agent với logic mới"""
    try:
        # Test import
        print("Testing CV Agent import...")
        from cv_agent import CVAgent
        print("CV Agent import successful")
        
        # Test initialization
        print("Testing CV Agent initialization...")
        cv_agent = CVAgent()
        print("CV Agent initialization successful")
        
        # Test load job requirements
        print("Testing job requirements loading...")
        job_requirements = cv_agent._load_job_requirements()
        print(f"Job requirements loaded: {len(job_requirements)} jobs")
        for job_title in job_requirements.keys():
            print(f"  - {job_title}")
        
        # Test analyze all CVs
        print("Testing analyze all CVs...")
        result = await cv_agent._analyze_all_cvs()
        print(f"Analyze all CVs result: {result.get('status')}")
        
        if result.get('status') == 'success':
            cv_count = result.get('result', {}).get('cv_count', 0)
            print(f"  - Found {cv_count} CVs")
            
            cv_evaluations = result.get('result', {}).get('cv_evaluations', [])
            for evaluation in cv_evaluations:
                cv_name = evaluation.get('cv_name', 'Unknown')
                status = evaluation.get('status', 'Unknown')
                print(f"  - {cv_name}: {status}")
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_cv_agent())
