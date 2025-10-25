# Multi-Agent HR System

Hệ thống Multi-Agent AI để phân tích HR với các agent chuyên biệt.

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐
│   Orchestrator  │ ← Phân tích intent và điều phối
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │   Router   │ ← Điều hướng đến agent phù hợp
    └─────┬─────┘
          │
    ┌─────▼─────────────────────────────────────┐
    │              Agents                       │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐     │
    │  │ Query   │ │   CV    │ │ Chart   │     │
    │  │ Agent   │ │ Agent   │ │ Agent   │     │
    │  └─────────┘ └─────────┘ └─────────┘     │
    │  ┌─────────┐ ┌─────────┐                 │
    │  │Analysis │ │   ...   │                 │
    │  │ Agent   │ │         │                 │
    │  └─────────┘ └─────────┘                 │
    └───────────────────────────────────────────┘
```

## 🤖 Các Agent

### 1. Orchestrator Agent
- **Chức năng**: Phân tích intent người dùng và điều phối các agent khác
- **Input**: Yêu cầu của người dùng
- **Output**: Kết quả từ agent phù hợp

### 2. Query Agent
- **Chức năng**: Truy vấn cơ sở dữ liệu thông qua MCP
- **Tools**: MCP Client, PostgreSQL tools
- **Input**: Câu hỏi về dữ liệu
- **Output**: Kết quả truy vấn có cấu trúc

### 3. CV Agent
- **Chức năng**: Phân tích CV và ứng viên
- **Tools**: PDF extraction, Gemini AI analysis
- **Input**: Yêu cầu phân tích CV
- **Output**: Kết quả phân tích CV và so sánh

### 4. Chart Agent
- **Chức năng**: Tạo biểu đồ và trực quan hóa dữ liệu
- **Tools**: Matplotlib, Pandas
- **Input**: Dữ liệu và yêu cầu tạo biểu đồ
- **Output**: File biểu đồ và phân tích

### 5. Analysis Agent
- **Chức năng**: Tổng hợp và phân tích kết quả từ các agent khác
- **Tools**: Gemini AI, Data analysis
- **Input**: Kết quả từ các agent khác
- **Output**: Báo cáo tổng hợp và insights

## 🚀 Cài đặt

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Cấu hình environment
Tạo file `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key
DB_LINK=your_database_connection_string
WREN_MCP_SERVER_PATH=D:/HR4/wren-engine/mcp-server/app/wren.py
```

### 3. Khởi tạo cấu hình
```bash
python config.py
```

## 🎯 Sử dụng

### 1. Chạy hệ thống chính
```bash
python main_agent_system.py
```

### 2. Test từng agent riêng lẻ
```bash
# Test Orchestrator
python orchestrator_agent.py

# Test Query Agent
python query_agent.py

# Test CV Agent
python cv_agent.py

# Test Chart Agent
python chart_agent.py

# Test Analysis Agent
python analysis_agent.py
```

### 3. Chạy MCP Server cho CV Agent
```bash
python cv_mcp_server.py
```

## 📋 Workflows

### 1. Query → Chart Workflow
```
User Input → Orchestrator → Query Agent → Chart Agent → Analysis Agent
```

### 2. CV Analysis Workflow
```
User Input → Orchestrator → CV Agent → Analysis Agent
```

### 3. Full Analysis Workflow
```
User Input → Orchestrator → All Agents (parallel) → Analysis Agent
```

## 🔧 Cấu hình

### Agent Settings
```python
# Trong config.py
AgentConfig.QUERY_AGENT = {
    "max_steps": 20,
    "timeout": 30
}

AgentConfig.CV_AGENT = {
    "quota_settings": {
        "max_requests_per_minute": 15,
        "delay_between_requests": 2
    }
}
```

### Workflow Settings
```python
WorkflowConfig.WORKFLOWS = {
    "query_then_chart": {
        "steps": ["query_agent", "chart_agent", "analysis_agent"],
        "timeout": 60
    }
}
```

## 📊 Ví dụ sử dụng

### 1. Truy vấn dữ liệu
```python
# Input: "Tìm nhân viên có lương cao nhất"
# Orchestrator → Query Agent
# Output: Danh sách nhân viên với lương cao nhất
```

### 2. Phân tích CV
```python
# Input: "Phân tích CV của ứng viên Python developer"
# Orchestrator → CV Agent
# Output: Phân tích chi tiết CV và đánh giá phù hợp
```

### 3. Tạo biểu đồ
```python
# Input: "Tạo biểu đồ thống kê nhân viên theo phòng ban"
# Orchestrator → Query Agent → Chart Agent
# Output: Biểu đồ trực quan và phân tích
```

## 🛠️ Mở rộng

### Thêm Agent mới
1. Tạo file `new_agent.py`
2. Implement class `NewAgent` với method `process()`
3. Thêm vào `orchestrator_agent.py`
4. Cập nhật `main_agent_system.py`

### Thêm Workflow mới
1. Cập nhật `WorkflowConfig.WORKFLOWS`
2. Thêm logic trong `process_workflow()`
3. Cập nhật menu trong `main_agent_system.py`

## 🐛 Troubleshooting

### Lỗi thường gặp

1. **Quota exceeded**
   - Giảm tần suất gọi API
   - Tăng delay giữa các request

2. **MCP connection failed**
   - Kiểm tra đường dẫn MCP server
   - Kiểm tra kết nối database

3. **File not found**
   - Kiểm tra đường dẫn CV folder
   - Kiểm tra quyền truy cập file

### Debug mode
```python
# Trong config.py
LoggingConfig.AGENT_LOG_LEVELS = {
    "orchestrator": "DEBUG",
    "query_agent": "DEBUG"
}
```

## 📈 Performance

### Optimization tips
1. Sử dụng cache cho kết quả thường xuyên
2. Chạy agents song song khi có thể
3. Giới hạn quota để tránh rate limit
4. Sử dụng connection pooling cho database

### Monitoring
- Logs được lưu trong `multi_agent_system.log`
- Conversation history được lưu trong memory
- Performance metrics có thể được thêm vào

## 🔒 Security

### API Keys
- Không commit API keys vào git
- Sử dụng environment variables
- Rotate keys định kỳ

### Data Privacy
- CV data được xử lý locally
- Không lưu trữ dữ liệu nhạy cảm
- Logs không chứa thông tin cá nhân

## 📚 API Reference

### OrchestratorAgent
```python
async def process(user_input: str) -> Dict[str, Any]
```

### QueryAgent
```python
async def process(user_input: str) -> Dict[str, Any]
```

### CVAgent
```python
async def process(user_input: str) -> Dict[str, Any]
```

### ChartAgent
```python
async def process(user_input: str, data: Any = None) -> Dict[str, Any]
```

### AnalysisAgent
```python
async def process(user_input: str, agent_results: List[Dict[str, Any]] = None) -> Dict[str, Any]
```

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.
