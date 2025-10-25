# Railway Deployment Setup

## Environment Variables cần thiết:

### 1. **GOOGLE_API_KEY** (Required)
- Lấy từ Google AI Studio: https://aistudio.google.com/
- Format: `AIza...`

### 2. **DB_LINK** (Required)
- PostgreSQL connection string
- Format: `postgresql://username:password@host:port/database`
- Ví dụ: `postgresql://user:pass@localhost:5432/hr_db`

### 3. **Railway Environment Variables** (Optional)
- `RAILWAY_ENVIRONMENT=production`
- `PORT=5000` (Railway tự động set)

## Cách setup trên Railway:

1. **Connect GitHub repository** to Railway
2. **Set Environment Variables** trong Railway dashboard:
   - `GOOGLE_API_KEY`: Your Google AI API key
   - `DB_LINK`: Your PostgreSQL connection string
3. **Deploy** - Railway sẽ tự động build và deploy

## Database Setup:

Railway cần PostgreSQL database với các bảng:
- `nhan_vien` (employees)
- `phong_ban` (departments) 
- `chuc_vu` (positions)
- `cong_viec` (tasks)
- `du_an` (projects)

## Troubleshooting:

- Nếu MCP Client fail → Check `DB_LINK` environment variable
- Nếu Google AI fail → Check `GOOGLE_API_KEY` environment variable
- Nếu build fail → Check Dockerfile và requirements.txt
