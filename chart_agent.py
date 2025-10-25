import asyncio
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Set matplotlib backend
matplotlib.use('Agg')

class ChartAgent:
    """
    Chart Agent - Tạo biểu đồ và trực quan hóa dữ liệu
    """
    
    def __init__(self):
        load_dotenv()
        self.GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not self.GEMINI_API_KEY:
            raise ValueError("⚠️ Thiếu GOOGLE_API_KEY trong .env")
        
        # LLM cho agent
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash-lite",
            google_api_key=self.GEMINI_API_KEY,
            temperature=0.2,
        )
        
        # Thư mục lưu biểu đồ
        self.chart_dir = Path("charts")
        self.chart_dir.mkdir(exist_ok=True)
    
    async def _analyze_chart_requirements(self, user_input: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Phân tích yêu cầu và dữ liệu để chọn loại biểu đồ phù hợp"""
        try:
            columns = data.get("columns", [])
            data_rows = data.get("data", [])
            
            # Chuẩn bị thông tin dữ liệu cho LLM
            data_info = f"Cột: {columns}\nDữ liệu mẫu: {data_rows[:3] if data_rows else []}"
            
            prompt = f"""
Bạn là chuyên gia phân tích dữ liệu và trực quan hóa. Hãy phân tích yêu cầu và dữ liệu để chọn loại biểu đồ phù hợp nhất.

Yêu cầu người dùng: "{user_input}"

Thông tin dữ liệu:
{data_info}

Hãy trả về JSON với format:
{{
    "chart_type": "bar|pie|line|scatter|histogram",
    "x_column": "tên cột cho trục X",
    "y_column": "tên cột cho trục Y", 
    "title": "tiêu đề biểu đồ",
    "reasoning": "lý do chọn loại biểu đồ này"
}}

Quy tắc chọn biểu đồ:
- bar: So sánh các danh mục, đếm số lượng
- pie: Phân bổ phần trăm, tỷ lệ
- line: Xu hướng theo thời gian
- scatter: Mối quan hệ giữa 2 biến số
- histogram: Phân phối của 1 biến số

Chọn loại biểu đồ phù hợp nhất với yêu cầu và dữ liệu.
"""
            
            response = await self.llm.ainvoke(prompt)
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            try:
                # Tìm JSON trong response
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    chart_config = json.loads(json_match.group(0))
                    return chart_config
            except json.JSONDecodeError:
                pass
            
            # Fallback: Phân tích đơn giản dựa trên từ khóa
            user_lower = user_input.lower()
            if "tròn" in user_lower or "pie" in user_lower or "phần trăm" in user_lower:
                return {
                    "chart_type": "pie",
                    "x_column": columns[0] if columns else "category",
                    "y_column": columns[1] if len(columns) > 1 else "value",
                    "title": "Biểu đồ tròn",
                    "reasoning": "Yêu cầu biểu đồ tròn"
                }
            elif "cột" in user_lower or "bar" in user_lower or "so sánh" in user_lower:
                return {
                    "chart_type": "bar",
                    "x_column": columns[0] if columns else "category", 
                    "y_column": columns[1] if len(columns) > 1 else "value",
                    "title": "Biểu đồ cột",
                    "reasoning": "Yêu cầu biểu đồ cột"
                }
            elif "đường" in user_lower or "line" in user_lower or "xu hướng" in user_lower:
                return {
                    "chart_type": "line",
                    "x_column": columns[0] if columns else "time",
                    "y_column": columns[1] if len(columns) > 1 else "value", 
                    "title": "Biểu đồ đường",
                    "reasoning": "Yêu cầu biểu đồ đường"
                }
            else:
                # Mặc định: bar chart
                return {
                    "chart_type": "bar",
                    "x_column": columns[0] if columns else "category",
                    "y_column": columns[1] if len(columns) > 1 else "value",
                    "title": "Biểu đồ cột",
                    "reasoning": "Mặc định chọn biểu đồ cột"
                }
                
        except Exception as e:
            print(f"⚠️ Lỗi phân tích yêu cầu biểu đồ: {e}")
            # Fallback đơn giản
            return {
                "chart_type": "bar",
                "x_column": data.get("columns", ["category"])[0],
                "y_column": data.get("columns", ["category", "value"])[1] if len(data.get("columns", [])) > 1 else "value",
                "title": "Biểu đồ",
                "reasoning": "Fallback mặc định"
            }
    
    async def _create_chart_by_type(self, chart_config: Dict[str, Any], data: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Tạo biểu đồ theo loại đã chọn"""
        try:
            chart_type = chart_config.get("chart_type", "bar")
            x_column = chart_config.get("x_column", "")
            y_column = chart_config.get("y_column", "")
            title = chart_config.get("title", "Biểu đồ")
            
            # Tìm index của các cột
            columns = data.get("columns", [])
            x_idx = columns.index(x_column) if x_column in columns else 0
            y_idx = columns.index(y_column) if y_column in columns else (1 if len(columns) > 1 else 0)
            
            # Chuẩn bị dữ liệu
            data_rows = data.get("data", [])
            if not data_rows:
                return {"error": "Không có dữ liệu để vẽ biểu đồ"}
            
            # Tạo biểu đồ theo loại
            if chart_type == "pie":
                return self._create_pie_chart(data_rows, x_idx, y_idx, title, user_input)
            elif chart_type == "line":
                return self._create_line_chart(data_rows, x_idx, y_idx, title, user_input)
            elif chart_type == "scatter":
                return self._create_scatter_chart(data_rows, x_idx, y_idx, title, user_input)
            elif chart_type == "histogram":
                return self._create_histogram_chart(data_rows, y_idx, title, user_input)
            else:  # bar chart (mặc định)
                return self._create_bar_chart(data_rows, x_idx, y_idx, title, user_input)
                
        except Exception as e:
            return {"error": f"Lỗi tạo biểu đồ: {str(e)}"}
    
    def _create_bar_chart(self, data_rows: List[List], x_idx: int, y_idx: int, title: str, user_input: str) -> Dict[str, Any]:
        """Tạo biểu đồ cột"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # Chuẩn bị dữ liệu
            x_data = [str(row[x_idx]) for row in data_rows]
            y_data = [float(row[y_idx]) if isinstance(row[y_idx], (int, float)) else 0 for row in data_rows]
            
            # Tạo biểu đồ
            plt.figure(figsize=(10, 6))
            bars = plt.bar(x_data, y_data, color='skyblue', edgecolor='navy', alpha=0.7)
            
            # Thêm giá trị trên mỗi cột
            for bar, value in zip(bars, y_data):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(y_data)*0.01, 
                        f'{value:.0f}', ha='center', va='bottom')
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Danh mục')
            plt.ylabel('Giá trị')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Lưu file
            chart_file = f"chart_{hash(user_input) % 1000000}.png"
            chart_path = self.chart_dir / chart_file
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "chart_file": str(chart_path),
                "chart_type": "bar",
                "title": title,
                "data_points": len(data_rows)
            }
            
        except Exception as e:
            return {"error": f"Lỗi tạo biểu đồ cột: {str(e)}"}
    
    def _create_pie_chart(self, data_rows: List[List], x_idx: int, y_idx: int, title: str, user_input: str) -> Dict[str, Any]:
        """Tạo biểu đồ tròn"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # Chuẩn bị dữ liệu
            labels = [str(row[x_idx]) for row in data_rows]
            values = [float(row[y_idx]) if isinstance(row[y_idx], (int, float)) else 0 for row in data_rows]
            
            # Tạo biểu đồ
            plt.figure(figsize=(10, 8))
            colors = plt.cm.Set3(range(len(labels)))
            wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', 
                                             colors=colors, startangle=90)
            
            # Tùy chỉnh text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.axis('equal')
            
            # Lưu file
            chart_file = f"chart_{hash(user_input) % 1000000}.png"
            chart_path = self.chart_dir / chart_file
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "chart_file": str(chart_path),
                "chart_type": "pie",
                "title": title,
                "data_points": len(data_rows)
            }
            
        except Exception as e:
            return {"error": f"Lỗi tạo biểu đồ tròn: {str(e)}"}
    
    def _create_line_chart(self, data_rows: List[List], x_idx: int, y_idx: int, title: str, user_input: str) -> Dict[str, Any]:
        """Tạo biểu đồ đường"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # Chuẩn bị dữ liệu
            x_data = [str(row[x_idx]) for row in data_rows]
            y_data = [float(row[y_idx]) if isinstance(row[y_idx], (int, float)) else 0 for row in data_rows]
            
            # Tạo biểu đồ
            plt.figure(figsize=(10, 6))
            plt.plot(x_data, y_data, marker='o', linewidth=2, markersize=6, color='blue')
            
            # Thêm điểm dữ liệu
            for i, (x, y) in enumerate(zip(x_data, y_data)):
                plt.annotate(f'{y:.0f}', (x, y), textcoords="offset points", 
                            xytext=(0,10), ha='center')
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Danh mục')
            plt.ylabel('Giá trị')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Lưu file
            chart_file = f"chart_{hash(user_input) % 1000000}.png"
            chart_path = self.chart_dir / chart_file
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "chart_file": str(chart_path),
                "chart_type": "line",
                "title": title,
                "data_points": len(data_rows)
            }
            
        except Exception as e:
            return {"error": f"Lỗi tạo biểu đồ đường: {str(e)}"}
    
    def _create_scatter_chart(self, data_rows: List[List], x_idx: int, y_idx: int, title: str, user_input: str) -> Dict[str, Any]:
        """Tạo biểu đồ phân tán"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # Chuẩn bị dữ liệu
            x_data = [float(row[x_idx]) if isinstance(row[x_idx], (int, float)) else 0 for row in data_rows]
            y_data = [float(row[y_idx]) if isinstance(row[y_idx], (int, float)) else 0 for row in data_rows]
            
            # Tạo biểu đồ
            plt.figure(figsize=(10, 6))
            plt.scatter(x_data, y_data, alpha=0.7, s=100, color='red')
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Lưu file
            chart_file = f"chart_{hash(user_input) % 1000000}.png"
            chart_path = self.chart_dir / chart_file
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "chart_file": str(chart_path),
                "chart_type": "scatter",
                "title": title,
                "data_points": len(data_rows)
            }
            
        except Exception as e:
            return {"error": f"Lỗi tạo biểu đồ phân tán: {str(e)}"}
    
    def _create_histogram_chart(self, data_rows: List[List], y_idx: int, title: str, user_input: str) -> Dict[str, Any]:
        """Tạo biểu đồ histogram"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # Chuẩn bị dữ liệu
            values = [float(row[y_idx]) if isinstance(row[y_idx], (int, float)) else 0 for row in data_rows]
            
            # Tạo biểu đồ
            plt.figure(figsize=(10, 6))
            plt.hist(values, bins=10, alpha=0.7, color='green', edgecolor='black')
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Giá trị')
            plt.ylabel('Tần suất')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Lưu file
            chart_file = f"chart_{hash(user_input) % 1000000}.png"
            chart_path = self.chart_dir / chart_file
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "chart_file": str(chart_path),
                "chart_type": "histogram",
                "title": title,
                "data_points": len(data_rows)
            }
            
        except Exception as e:
            return {"error": f"Lỗi tạo biểu đồ histogram: {str(e)}"}
    
    def _normalize_data(self, data: Any) -> Dict[str, Any]:
        """Chuẩn hóa dữ liệu về format {columns, data}"""
        if isinstance(data, dict) and "columns" in data and "data" in data:
            return data
        
        # Convert pandas DataFrame
        if isinstance(data, pd.DataFrame):
            return {
                "columns": data.columns.tolist(),
                "data": data.values.tolist()
            }
        
        # Convert list of dicts
        if isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            return {
                "columns": df.columns.tolist(),
                "data": df.values.tolist()
            }
        
        # Convert dict of arrays
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            df = pd.DataFrame(data)
            return {
                "columns": df.columns.tolist(),
                "data": df.values.tolist()
            }
        
        # Handle string data from Query Agent
        if isinstance(data, str):
            try:
                # Try to extract data from markdown table or text
                import re
                
                # Look for markdown table pattern - improved regex
                table_match = re.search(r'\|.*\|.*\n\|.*\|.*\n((?:\|.*\|.*\n?)*)', data, re.MULTILINE)
                if table_match:
                    table_text = table_match.group(0)
                    lines = [line.strip() for line in table_text.split('\n') if line.strip() and '|' in line]
                    
                    if len(lines) >= 2:
                        # Parse header
                        header_line = lines[0]
                        columns = [col.strip() for col in header_line.split('|') if col.strip()]
                        
                        # Parse data rows
                        data_rows = []
                        for line in lines[1:]:
                            if '|' in line:
                                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                                if len(row) == len(columns):
                                    data_rows.append(row)
                        
                        if data_rows:
                            return {
                                "columns": columns,
                                "data": data_rows
                            }
                
                # Alternative: Look for table pattern with different separators
                table_pattern = re.search(r'\|([^|]+)\|([^|]+)\|\s*\n\|[-\s|]+\|\s*\n((?:\|[^|]+\|[^|]+\|\s*\n?)*)', data, re.MULTILINE)
                if table_pattern:
                    header1 = table_pattern.group(1).strip()
                    header2 = table_pattern.group(2).strip()
                    table_data = table_pattern.group(3)
                    
                    columns = [header1, header2]
                    data_rows = []
                    
                    for line in table_data.strip().split('\n'):
                        if '|' in line:
                            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                            if len(cells) >= 2:
                                data_rows.append(cells[:2])  # Take only first 2 columns
                    
                    if data_rows:
                        return {
                            "columns": columns,
                            "data": data_rows
                        }
                
                # Look for JSON-like data in the text
                json_match = re.search(r'\[.*?\]', data, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group())
                        if isinstance(json_data, list) and json_data:
                            df = pd.DataFrame(json_data)
                            return {
                                "columns": df.columns.tolist(),
                                "data": df.values.tolist()
                            }
                    except:
                        pass
                
                # Try to extract structured data from text
                # Look for patterns like "Phòng Nhân sự: 9 nhân viên"
                pattern = r'([^:]+):\s*(\d+)'
                matches = re.findall(pattern, data)
                if matches:
                    columns = ["PhongBan", "SoLuong"]
                    data_rows = [[match[0].strip(), int(match[1])] for match in matches]
                    return {
                        "columns": columns,
                        "data": data_rows
                    }
                
            except Exception as e:
                print(f"⚠️ Error parsing string data: {e}")
        
        return {"columns": [], "data": []}
    
    def _create_chart(self, data: Dict[str, Any], chart_type: str = "bar", title: str = "", 
                     x_col: str = None, y_col: str = None) -> Dict[str, Any]:
        """Tạo biểu đồ từ dữ liệu"""
        try:
            if not data.get("data") or not data.get("columns"):
                return {"error": "No data to visualize"}
            
            df = pd.DataFrame(data["data"], columns=data["columns"])
            
            if df.empty:
                return {"error": "Empty dataframe"}
            
            # Tự động xác định x_col và y_col nếu không được cung cấp
            if not x_col and not y_col:
                if len(df.columns) >= 2:
                    x_col = df.columns[0]
                    y_col = df.columns[1]
                else:
                    x_col = df.columns[0]
                    y_col = df.columns[0]
            
            # Tạo biểu đồ
            plt.figure(figsize=(12, 8))
            
            if chart_type == "bar":
                if x_col and y_col and x_col != y_col:
                    plt.bar(df[x_col], df[y_col])
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                else:
                    plt.bar(range(len(df)), df[x_col])
                    plt.xlabel("Index")
                    plt.ylabel(x_col)
            
            elif chart_type == "pie":
                if x_col and y_col and x_col != y_col:
                    plt.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%')
                else:
                    plt.pie(df[x_col], autopct='%1.1f%%')
            
            elif chart_type == "line":
                if x_col and y_col and x_col != y_col:
                    plt.plot(df[x_col], df[y_col], marker='o')
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                else:
                    plt.plot(df[x_col], marker='o')
                    plt.xlabel("Index")
                    plt.ylabel(x_col)
            
            elif chart_type == "scatter":
                if x_col and y_col and x_col != y_col:
                    plt.scatter(df[x_col], df[y_col])
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                else:
                    return {"error": "Scatter plot requires two different columns"}
            
            elif chart_type == "histogram":
                plt.hist(df[x_col], bins=20, alpha=0.7)
                plt.xlabel(x_col)
                plt.ylabel("Frequency")
            
            else:
                return {"error": f"Unsupported chart type: {chart_type}"}
            
            # Set title
            if title:
                plt.title(title)
            else:
                plt.title(f"{chart_type.title()} Chart")
            
            plt.tight_layout()
            
            # Save chart
            chart_file = self.chart_dir / f"chart_{hash(json.dumps(data, ensure_ascii=False)) % 1000000}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "chart_file": str(chart_file),
                "chart_type": chart_type,
                "title": title or f"{chart_type.title()} Chart",
                "data_shape": {
                    "rows": len(df),
                    "columns": len(df.columns)
                }
            }
            
        except Exception as e:
            return {"error": f"Chart creation failed: {str(e)}"}
    
    def _suggest_chart_type(self, data: Dict[str, Any]) -> str:
        """Gợi ý loại biểu đồ phù hợp dựa trên dữ liệu"""
        if not data.get("data") or not data.get("columns"):
            return "bar"
        
        df = pd.DataFrame(data["data"], columns=data["columns"])
        
        # Phân tích dữ liệu để gợi ý
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            return "scatter"
        elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            return "bar"
        elif len(numeric_cols) >= 1:
            return "histogram"
        else:
            return "bar"
    
    async def _plan_chart(self, data: Dict[str, Any], intent: str = "") -> Dict[str, Any]:
        """Lập kế hoạch tạo biểu đồ"""
        try:
            normalized_data = self._normalize_data(data)
            
            if not normalized_data.get("data"):
                return {
                    "error": "No data available for charting",
                    "plan": None
                }
            
            df = pd.DataFrame(normalized_data["data"], columns=normalized_data["columns"])
            
            # Phân tích dữ liệu
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Gợi ý loại biểu đồ
            suggested_type = self._suggest_chart_type(normalized_data)
            
            # Tạo kế hoạch
            plan = {
                "chart_type": suggested_type,
                "title": intent or f"Biểu đồ {suggested_type}",
                "x_column": categorical_cols[0] if categorical_cols else numeric_cols[0],
                "y_column": numeric_cols[0] if numeric_cols else None,
                "data_analysis": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "numeric_columns": numeric_cols,
                    "categorical_columns": categorical_cols
                },
                "recommendations": []
            }
            
            # Thêm gợi ý
            if len(numeric_cols) >= 2:
                plan["recommendations"].append("Có thể tạo scatter plot để xem mối quan hệ giữa các biến số")
            if categorical_cols and numeric_cols:
                plan["recommendations"].append("Có thể tạo bar chart để so sánh các danh mục")
            if len(numeric_cols) >= 1:
                plan["recommendations"].append("Có thể tạo histogram để phân phối dữ liệu")
            
            return {
                "success": True,
                "plan": plan
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Planning failed: {str(e)}"
            }
    
    async def _analyze_chart_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Phân tích dữ liệu biểu đồ"""
        try:
            normalized_data = self._normalize_data(data)
            
            if not normalized_data.get("data"):
                return {
                    "success": False,
                    "error": "No data to analyze"
                }
            
            df = pd.DataFrame(normalized_data["data"], columns=normalized_data["columns"])
            
            analysis = {
                "basic_stats": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "columns": df.columns.tolist()
                },
                "numeric_analysis": {},
                "categorical_analysis": {}
            }
            
            # Phân tích cột số
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                analysis["numeric_analysis"][col] = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max()
                }
            
            # Phân tích cột phân loại
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                analysis["categorical_analysis"][col] = {
                    "unique_values": df[col].nunique(),
                    "most_common": value_counts.head(3).to_dict(),
                    "missing_values": df[col].isnull().sum()
                }
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }
    
    async def process(self, user_input: str, data: Any = None) -> Dict[str, Any]:
        """
        Xử lý yêu cầu tạo biểu đồ
        """
        try:
            print(f"Chart Agent: Xu ly yeu cau")
            
            # Nếu không có dữ liệu, trả về hướng dẫn
            if data is None:
                return {
                    "agent": "chart_agent",
                    "status": "info",
                    "result": {
                        "message": "Chart Agent sẵn sàng tạo biểu đồ",
                        "supported_types": ["bar", "pie", "line", "scatter", "histogram"],
                        "usage": "Cung cấp dữ liệu để tạo biểu đồ"
                    }
                }
            
            print(f"Chart Agent: Nhan du lieu tu Query Agent")
            print(f"Data type: {type(data)}")
            if isinstance(data, str):
                print(f"Data preview: {len(data)} characters")
            else:
                print(f"Data: {data}")
            
            # Chuẩn hóa dữ liệu
            normalized_data = self._normalize_data(data)
            
            print(f"Normalized data columns: {len(normalized_data.get('columns', []))}")
            print(f"Normalized data rows: {len(normalized_data.get('data', []))}")
            
            if not normalized_data.get("data"):
                return {
                    "agent": "chart_agent",
                    "status": "error",
                    "error": f"Không thể parse dữ liệu từ Query Agent. Data type: {type(data)}"
                }
            
            # Phân tích yêu cầu và chọn loại biểu đồ phù hợp
            print(f"Chart Agent: Phân tích yêu cầu và chọn loại biểu đồ...")
            chart_config = await self._analyze_chart_requirements(user_input, normalized_data)
            
            print(f"Chart Agent: Đã chọn biểu đồ {chart_config['chart_type']} - {chart_config['reasoning']}")
            
            # Tạo biểu đồ theo loại đã chọn
            chart_result = await self._create_chart_by_type(chart_config, normalized_data, user_input)
            
            if "error" in chart_result:
                return {
                    "agent": "chart_agent",
                    "status": "error",
                    "error": chart_result["error"]
                }
            
            return {
                "agent": "chart_agent",
                "status": "success",
                "result": {
                    "chart_info": chart_result,
                    "chart_config": chart_config,
                    "data_summary": f"Đã tạo biểu đồ {chart_config['chart_type']} với {len(normalized_data.get('data', []))} điểm dữ liệu"
                }
            }
            
        except Exception as e:
            return {
                "agent": "chart_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def create_multiple_charts(self, data: Dict[str, Any], chart_types: List[str] = None) -> Dict[str, Any]:
        """Tạo nhiều biểu đồ từ cùng một dữ liệu"""
        try:
            if chart_types is None:
                chart_types = ["bar", "pie", "line"]
            
            normalized_data = self._normalize_data(data)
            
            if not normalized_data.get("data"):
                return {
                    "agent": "chart_agent",
                    "status": "error",
                    "error": "Không có dữ liệu để tạo biểu đồ"
                }
            
            results = []
            
            for chart_type in chart_types:
                chart_result = self._create_chart(
                    normalized_data,
                    chart_type=chart_type,
                    title=f"Biểu đồ {chart_type}"
                )
                
                if "error" not in chart_result:
                    results.append(chart_result)
            
            return {
                "agent": "chart_agent",
                "status": "success",
                "result": {
                    "total_charts": len(results),
                    "charts": results
                }
            }
            
        except Exception as e:
            return {
                "agent": "chart_agent",
                "status": "error",
                "error": str(e)
            }

# Test function
async def test_chart_agent():
    agent = ChartAgent()
    
    # Test data
    test_data = {
        "columns": ["PhongBan", "SoLuong"],
        "data": [
            ["IT", 15],
            ["HR", 8],
            ["Finance", 12],
            ["Marketing", 10]
        ]
    }
    
    test_cases = [
        ("Tạo biểu đồ cột cho dữ liệu phòng ban", test_data),
        ("Tạo biểu đồ tròn cho thống kê nhân viên", test_data),
        ("Tạo biểu đồ đường cho xu hướng", test_data)
    ]
    
    for test_input, data in test_cases:
        print(f"\n{'='*50}")
        print(f"Test: {test_input}")
        result = await agent.process(test_input, data)
        print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_chart_agent())
