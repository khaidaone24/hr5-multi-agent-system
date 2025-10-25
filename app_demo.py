#!/usr/bin/env python3
"""
Demo version of Multi-Agent HR System - No external dependencies required
"""

import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hr5-multi-agent-system'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('charts', exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_request():
    """Xử lý yêu cầu từ người dùng - Demo version"""
    try:
        data = request.get_json()
        user_input = data.get('input', '').strip()
        uploaded_files = data.get('files', [])
        
        if not user_input:
            return jsonify({'error': 'Vui lòng nhập yêu cầu'}), 400
        
        # Demo response
        demo_response = {
            'orchestrator_analysis': {
                'intent_analysis': {
                    'primary_intent': 'demo_query',
                    'confidence': 0.95,
                    'required_agents': ['query_agent']
                }
            },
            'agent_results': [
                {
                    'step': 1,
                    'agent': 'query_agent',
                    'status': 'completed',
                    'result': {
                        'final_answer': f'🎯 **Demo Response**: Bạn đã hỏi: "{user_input}"\n\n📊 **Thống kê**:\n- Từ khóa: {len(user_input.split())} từ\n- Files đã upload: {len(uploaded_files)}\n- Thời gian: {datetime.now().strftime("%H:%M:%S")}\n\n💡 **Ghi chú**: Đây là phiên bản demo. Để sử dụng đầy đủ tính năng, cần cấu hình GOOGLE_API_KEY.'
                    }
                }
            ],
            'analysis_result': {
                'result': {
                    'markdown': '### 🚀 **Multi-Agent HR System Demo**\n\nHệ thống đang chạy ở chế độ demo. Để sử dụng đầy đủ tính năng AI, vui lòng cấu hình biến môi trường `GOOGLE_API_KEY`.'
                }
            }
        }
        
        return jsonify(demo_response)
            
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload file PDF - Demo version"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        if file and allowed_file(file.filename):
            # Demo response - không thực sự lưu file
            return jsonify({
                'success': True,
                'filename': f'demo_{file.filename}',
                'original_name': file.filename,
                'message': 'File đã được upload thành công (Demo mode)'
            })
        else:
            return jsonify({'error': 'Chỉ cho phép file PDF'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Lỗi upload: {str(e)}'}), 500

@app.route('/api/charts/<filename>')
def get_chart(filename):
    """Lấy file biểu đồ - Demo version"""
    return jsonify({'error': 'Demo mode - Chart generation disabled'}), 404

@app.route('/api/status')
def get_status():
    """Kiểm tra trạng thái hệ thống"""
    return jsonify({
        'status': 'running',
        'mode': 'demo',
        'agents': [
            'Demo Agent'
        ],
        'timestamp': datetime.now().isoformat(),
        'message': '🚀 Multi-Agent HR System - Demo Mode'
    })

if __name__ == '__main__':
    # Get port from environment variable (for deployment platforms)
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Starting Multi-Agent HR System - DEMO MODE...")
    print(f"📱 Open browser and go to: http://localhost:{port}")
    print("⚠️  Note: This is a demo version. Full features require GOOGLE_API_KEY configuration.")
    
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False, threaded=False)
