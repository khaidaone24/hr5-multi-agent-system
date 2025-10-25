#!/bin/bash
echo "🚀 Starting HR System with Database..."

# Start PostgreSQL and HR App with Docker Compose
docker-compose up --build

echo "✅ HR System is running with database!"
echo "📱 Web interface: http://localhost:5000"
echo "🗄️ Database: PostgreSQL on localhost:5432"
