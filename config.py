"""
Cấu hình cho Multi-Agent HR System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Cấu hình chính của hệ thống"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DB_LINK = os.getenv("DB_LINK")
    
    # Model settings
    LLM_MODEL = "models/gemini-2.0-flash-lite"
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 4000
    
    # MCP Server paths
    WREN_MCP_SERVER_PATH = os.getenv("WREN_MCP_SERVER_PATH", "D:/HR4/wren-engine/mcp-server/app/wren.py")
    CV_MCP_SERVER_PATH = "cv_mcp_server.py"
    
    # File paths
    CV_FOLDER = Path("D:/HR4/PDF")
    JOB_REQUIREMENTS_FILE = Path("D:/HR4/job_requirements/job_requirements.xlsx")
    CHART_OUTPUT_DIR = Path("charts")
    CACHE_DIR = Path("cache")
    
    # Quota settings
    MAX_REQUESTS_PER_MINUTE = 15
    RATE_LIMIT_DELAY = 2  # seconds
    
    # Agent settings
    MAX_STEPS = 20
    TIMEOUT_SECONDS = 30
    
    # Chart settings
    CHART_DPI = 300
    CHART_FIGSIZE = (12, 8)
    
    # Database settings
    DB_POOL_SIZE = 5
    DB_TIMEOUT = 30
    
    @classmethod
    def validate(cls):
        """Kiểm tra cấu hình"""
        errors = []
        
        if not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is required")
        
        if not cls.DB_LINK:
            errors.append("DB_LINK is required")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True
    
    @classmethod
    def setup_directories(cls):
        """Tạo các thư mục cần thiết"""
        cls.CHART_OUTPUT_DIR.mkdir(exist_ok=True)
        cls.CACHE_DIR.mkdir(exist_ok=True)
        
        print(f" Created directories: {cls.CHART_OUTPUT_DIR}, {cls.CACHE_DIR}")

# Agent configurations
class AgentConfig:
    """Cấu hình cho từng agent"""
    
    # Orchestrator
    ORCHESTRATOR = {
        "intent_patterns": {
            "query": {
                "keywords": ["truy vấn", "query", "sql", "database", "dữ liệu", "bảng", "nhân viên", "phòng ban", "chức vụ", "công việc"],
                "agent": "query_agent",
                "description": "Truy vấn cơ sở dữ liệu"
            },
            "cv_analysis": {
                "keywords": ["cv", "hồ sơ", "ứng viên", "phân tích cv", "so sánh cv", "đánh giá cv", "tìm ứng viên"],
                "agent": "cv_agent", 
                "description": "Phân tích CV và ứng viên"
            },
            "chart": {
                "keywords": ["biểu đồ", "chart", "đồ thị", "visualization", "trực quan hóa", "thống kê", "báo cáo"],
                "agent": "chart_agent",
                "description": "Tạo biểu đồ và trực quan hóa dữ liệu"
            },
            "analysis": {
                "keywords": ["phân tích", "analysis", "tổng hợp", "báo cáo", "kết luận", "đánh giá tổng thể"],
                "agent": "analysis_agent",
                "description": "Phân tích và tổng hợp kết quả"
            }
        }
    }
    
    # Query Agent
    QUERY_AGENT = {
        "mcp_config": {
            "mcpServers": {
                "postgres": {
                    "command": "uv",
                    "args": ["run", "postgres-mcp", "--access-mode=unrestricted"],
                    "env": {"DATABASE_URI": Config.DB_LINK}
                }
            }
        },
        "max_steps": 20,
        "timeout": 30
    }
    
    # CV Agent
    CV_AGENT = {
        "extraction_settings": {
            "max_text_length": 8000,
            "key_info_limit": 10
        },
        "gemini_settings": {
            "model": "gemini-2.0-flash-exp",
            "temperature": 0.3,
            "max_output_tokens": 500,
            "top_p": 0.8
        },
        "quota_settings": {
            "max_requests_per_minute": 15,
            "delay_between_requests": 2
        }
    }
    
    # Chart Agent
    CHART_AGENT = {
        "supported_types": ["bar", "pie", "line", "scatter", "histogram"],
        "default_settings": {
            "figsize": (12, 8),
            "dpi": 300,
            "bbox_inches": "tight"
        },
        "auto_detection": {
            "numeric_columns_threshold": 2,
            "categorical_columns_threshold": 1
        }
    }
    
    # Analysis Agent
    ANALYSIS_AGENT = {
        "ai_analysis": {
            "model": "models/gemini-2.0-flash-lite",
            "temperature": 0.3,
            "max_tokens": 2000
        },
        "data_quality": {
            "missing_threshold": 0.1,
            "duplicate_threshold": 0.05
        }
    }

# Workflow configurations
class WorkflowConfig:
    """Cấu hình cho các workflow"""
    
    WORKFLOWS = {
        "query_then_chart": {
            "steps": ["query_agent", "chart_agent", "analysis_agent"],
            "description": "Query dữ liệu rồi tạo biểu đồ"
        },
        "cv_analysis": {
            "steps": ["cv_agent", "analysis_agent"],
            "description": "Phân tích CV và ứng viên"
        },
        "full_analysis": {
            "steps": ["query_agent", "cv_agent", "chart_agent", "analysis_agent"],
            "description": "Phân tích toàn diện với tất cả agents"
        }
    }
    
    # Timeout settings for workflows
    WORKFLOW_TIMEOUTS = {
        "query_then_chart": 60,
        "cv_analysis": 120,
        "full_analysis": 180
    }

# Logging configuration
class LoggingConfig:
    """Cấu hình logging"""
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "multi_agent_system.log"
    
    # Agent-specific logging
    AGENT_LOG_LEVELS = {
        "orchestrator": "DEBUG",
        "query_agent": "INFO", 
        "cv_agent": "INFO",
        "chart_agent": "INFO",
        "analysis_agent": "DEBUG"
    }

# Error handling configuration
class ErrorConfig:
    """Cấu hình xử lý lỗi"""
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    EXPONENTIAL_BACKOFF = True
    
    # Error types and handling
    ERROR_HANDLING = {
        "quota_exceeded": {
            "action": "wait",
            "delay": 60
        },
        "connection_error": {
            "action": "retry",
            "max_retries": 3
        },
        "file_not_found": {
            "action": "skip",
            "log_level": "WARNING"
        }
    }

# Initialize configuration
def initialize_config():
    """Khởi tạo cấu hình"""
    try:
        Config.validate()
        Config.setup_directories()
        print(" Configuration initialized successfully")
        return True
    except Exception as e:
        print(f" Configuration initialization failed: {e}")
        return False

if __name__ == "__main__":
    initialize_config()



