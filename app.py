#!/usr/bin/env python3
"""
Web Interface cho Multi-Agent HR System
"""

import asyncio
import json
import os
import sys
import traceback
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
try:
    import numpy as np  # type: ignore
except Exception:
    np = None

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from main_agent_system import MultiAgentSystem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hr5-multi-agent-system'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('charts', exist_ok=True)

# Thread-safe system instances storage
system_instances = {}
system_lock = threading.Lock()

def get_or_create_system():
    """Get or create system instance for current session"""
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    with system_lock:
        if session_id not in system_instances:
            print(f"🆕 Creating new system instance for session: {session_id}")
            system_instances[session_id] = MultiAgentSystem()
        return system_instances[session_id]

def cleanup_old_sessions():
    """Clean up old session instances to prevent memory leaks"""
    import time
    current_time = time.time()
    
    with system_lock:
        # Keep only last 10 sessions to prevent memory issues
        if len(system_instances) > 10:
            # Remove oldest sessions (simple FIFO)
            oldest_sessions = list(system_instances.keys())[:-10]
            for session_id in oldest_sessions:
                print(f"🧹 Cleaning up old session: {session_id}")
                del system_instances[session_id]

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/api/status')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/health')
def health():
    """Health check endpoint for Docker"""
    return "OK", 200

@app.route('/api/sessions')
def get_sessions():
    """Get current session information"""
    with system_lock:
        return jsonify({
            "active_sessions": len(system_instances),
            "session_ids": list(system_instances.keys())
        })

@app.route('/api/process', methods=['POST'])
def process_request():
    """Xử lý yêu cầu từ người dùng"""
    try:
        print(f"[DEBUG] Starting process_request")
        data = request.get_json()
        user_input = data.get('input', '').strip()
        uploaded_files = data.get('files', [])
        
        print(f"[DEBUG] User input: {user_input}")
        print(f"[DEBUG] Uploaded files: {uploaded_files}")
        
        if not user_input:
            return jsonify({'error': 'Vui long nhap yeu cau'}), 400
        
        # Get session-specific system instance
        system = get_or_create_system()
        session_id = session.get('session_id', 'unknown')
        print(f"[DEBUG] Using session: {session_id}")
        
        # Clean up old sessions periodically
        cleanup_old_sessions()
        
        # Process with asyncio.run to avoid closed loop issues
        try:
            print(f"[DEBUG] Calling system.process_single_request")
            result = asyncio.run(system.process_single_request(user_input, uploaded_files))
            print(f"[DEBUG] Got result from system: {type(result)}")
            
            # Convert numpy/pandas-like types for JSON serialization
            def convert_numpy_types(obj):
                try:
                    # Handle Decimal objects first (from database)
                    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Decimal':
                        return float(obj)
                    
                    # Handle NaN, Infinity values
                    if isinstance(obj, float):
                        if str(obj).lower() in ['nan', 'inf', '-inf', 'infinity', '-infinity']:
                            return None
                    
                    # Handle datetime objects
                    if hasattr(obj, 'isoformat'):
                        return obj.isoformat()
                    
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
                        # Handle numpy NaN/Inf
                        if isinstance(obj, np.floating):
                            if np.isnan(obj) or np.isinf(obj):
                                return None
                    
                    # pandas objects
                    try:
                        import pandas as pd
                        if isinstance(obj, pd.Series):
                            return obj.tolist()
                        if isinstance(obj, pd.DataFrame):
                            return obj.to_dict('records')
                    except ImportError:
                        pass
                    
                    # built-ins and containers
                    if hasattr(obj, 'item'):
                        return obj.item()
                    if isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple, set)):
                        return [convert_numpy_types(item) for item in obj]
                    return obj
                except Exception as e:
                    print(f"[DEBUG] Error converting object {type(obj)}: {e}")
                    # last resort stringify
                    return str(obj)
            
            print(f"[DEBUG] Converting numpy types")
            print(f"[DEBUG] Result type: {type(result)}")
            print(f"[DEBUG] Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            result_clean = convert_numpy_types(result)
            print(f"[DEBUG] Clean result type: {type(result_clean)}")
            print(f"[DEBUG] Clean result keys: {result_clean.keys() if isinstance(result_clean, dict) else 'Not a dict'}")
            
            print(f"[DEBUG] Returning clean result")
            return jsonify(result_clean)
            
        except Exception as e:
            print(f"[DEBUG] Error in asyncio.run: {e}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            raise e
            
        finally:
            pass
            
    except Exception as e:
        return jsonify({'error': f'Loi xu ly: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload file PDF"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Khong co file duoc chon'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Khong co file duoc chon'}), 400
        
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
                'message': 'File da duoc upload thanh cong'
            })
        else:
            return jsonify({'error': 'Chi cho phep file PDF'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Loi upload: {str(e)}'}), 500

@app.route('/api/charts/<filename>')
def get_chart(filename):
    """Lấy file biểu đồ"""
    try:
        chart_path = os.path.join('charts', filename)
        if os.path.exists(chart_path):
            return send_file(chart_path, mimetype='image/png')
        else:
            return jsonify({'error': 'File bieu do khong ton tai'}), 404
    except Exception as e:
        return jsonify({'error': f'Loi tai bieu do: {str(e)}'}), 500

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

if __name__ == '__main__':
    # Get port from environment variable (for deployment platforms)
    port = int(os.environ.get('PORT', 5000))
    
    print("Starting Multi-Agent HR System Web Interface...")
    print("FULL VERSION - Not Demo Mode")
    print(f"Open browser and go to: http://localhost:{port}")
    # Chạy 1 process/1 loop ổn định để tránh 'Event loop is closed'
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False, threaded=False)
