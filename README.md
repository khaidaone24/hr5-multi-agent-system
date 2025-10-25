# Multi-Agent HR System

Hệ thống AI Agent thông minh cho quản lý nhân sự với giao diện web hiện đại.

## 🚀 Demo Online

### Các cách deploy miễn phí:

#### 1. Railway (Khuyến nghị - Dễ nhất)
1. Truy cập [Railway.app](https://railway.app)
2. Đăng nhập bằng GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Chọn repository này
5. Railway sẽ tự động deploy và cung cấp link public

#### 2. Render
1. Truy cập [Render.com](https://render.com)
2. Đăng nhập bằng GitHub
3. Click "New" → "Web Service"
4. Connect GitHub repository
5. Chọn "Free" plan
6. Deploy!

#### 3. Heroku
1. Tạo tài khoản [Heroku](https://heroku.com)
2. Cài Heroku CLI
3. Chạy lệnh:
```bash
heroku create your-app-name
git push heroku main
```

#### 4. Vercel
1. Truy cập [Vercel.com](https://vercel.com)
2. Import GitHub repository
3. Deploy tự động

## 🛠️ Chạy Local

```bash
# Clone repository
git clone <your-repo-url>
cd HR5

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy ứng dụng
python app.py
```

Truy cập: http://localhost:5000

## 📋 Tính năng

- **Orchestrator Agent**: Điều phối các agent khác
- **Query Agent**: Xử lý truy vấn dữ liệu
- **CV Agent**: Phân tích CV và ứng viên
- **Chart Agent**: Tạo biểu đồ thống kê
- **Analysis Agent**: Phân tích dữ liệu nâng cao

## 🔧 Cấu hình

Tạo file `.env` với các biến môi trường:

```env
GOOGLE_API_KEY=your_google_api_key
FLASK_ENV=production
```

## 📁 Cấu trúc dự án

```
HR5/
├── app.py                 # Flask web interface
├── main_agent_system.py  # Multi-agent system
├── orchestrator_agent.py  # Orchestrator agent
├── query_agent.py        # Query processing agent
├── cv_agent.py          # CV analysis agent
├── chart_agent.py       # Chart generation agent
├── analysis_agent.py    # Data analysis agent
├── templates/
│   └── index.html       # Web interface
└── requirements.txt     # Dependencies
```

## 🚀 Deployment Status

- ✅ Railway ready
- ✅ Render ready  
- ✅ Heroku ready
- ✅ Vercel ready
- ✅ Docker ready

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy tạo issue trên GitHub repository.
