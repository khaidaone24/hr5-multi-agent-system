#!/usr/bin/env python3
"""
Vercel serverless function for Multi-Agent HR System
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
try:
    import numpy as np  # type: ignore
except Exception:
    np = None

# Add current directory to path
sys.path.append(str(Path(__file__).parent.parent))

from main_agent_system import MultiAgentSystem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hr5-multi-agent-system'
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('/tmp/charts', exist_ok=True)

# Initialize Multi-Agent System
system = MultiAgentSystem()

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_request():
    """Xử lý yêu cầu từ người dùng"""
    try:
        data = request.get_json()
        user_input = data.get('input', '').strip()
        uploaded_files = data.get('files', [])
        
        if not user_input:
            return jsonify({'error': 'Vui lòng nhập yêu cầu'}), 400
        
        # Process with asyncio.run to avoid closed loop issues
        try:
            result = asyncio.run(system.process_single_request(user_input, uploaded_files))
            
            # Convert numpy/pandas-like types for JSON serialization
            def convert_numpy_types(obj):
                try:
                    # numpy arrays and scalars
                    if np is not None:
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, (np.generic,)):
                            return obj.item()
                        if isinstance(obj, (np.integer, np.floating, np.bool_)):
                            return obj.item()
                        if isinstance(obj, (np.datetime64,)):
                            return str(obj)
                        if isinstance(obj, (np.dtype,)):
                            return str(obj)
                    # built-ins and containers
                    if hasattr(obj, 'item'):
                        return obj.item()
                    if isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple, set)):
                        return [convert_numpy_types(item) for item in obj]
                    return obj
                except Exception:
                    # last resort stringify
                    return str(obj)
            
            result_clean = convert_numpy_types(result)
            return jsonify(result_clean)
            
        finally:
            pass
            
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload file PDF"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'filename': unique_filename,
                'original_name': filename,
                'message': 'File đã được upload thành công'
            })
        else:
            return jsonify({'error': 'Chỉ cho phép file PDF'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Lỗi upload: {str(e)}'}), 500

@app.route('/api/charts/<filename>')
def get_chart(filename):
    """Lấy file biểu đồ"""
    try:
        chart_path = os.path.join('/tmp/charts', filename)
        if os.path.exists(chart_path):
            return send_file(chart_path, mimetype='image/png')
        else:
            return jsonify({'error': 'File biểu đồ không tồn tại'}), 404
    except Exception as e:
        return jsonify({'error': f'Lỗi tải biểu đồ: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    """Kiểm tra trạng thái hệ thống"""
    return jsonify({
        'status': 'running',
        'agents': [
            'Orchestrator',
            'Query Agent', 
            'CV Agent',
            'Chart Agent',
            'Analysis Agent'
        ],
        'timestamp': datetime.now().isoformat()
    })

# Vercel serverless handler
def handler(request):
    return app(request.environ, lambda *args: None)
