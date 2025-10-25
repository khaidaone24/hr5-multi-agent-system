import asyncio
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
import ast
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
            raise ValueError(" Thiếu GOOGLE_API_KEY trong .env")
        
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
            
            # Tự động phân tích dữ liệu trước
            data_analysis = self._analyze_data_structure(data)
            
            # Chuẩn bị thông tin dữ liệu cho LLM với xử lý tốt hơn
            cleaned_data_info = self._clean_data_for_analysis(columns, data_rows)
            
            prompt = f"""
Bạn là chuyên gia phân tích dữ liệu và trực quan hóa. Hãy phân tích yêu cầu và dữ liệu để chọn loại biểu đồ phù hợp nhất.

Yêu cầu người dùng: "{user_input}"

Thông tin dữ liệu:
{cleaned_data_info}

Phân tích tự động:
- Cột có sẵn: {data_analysis.get('available_columns', [])}
- Cột số: {data_analysis.get('numeric_columns', [])}
- Cột phân loại: {data_analysis.get('categorical_columns', [])}
- Gợi ý X: {data_analysis.get('suggested_x', '')}
- Gợi ý Y: {data_analysis.get('suggested_y', '')}

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
                    # Validate và làm sạch config
                    return self._validate_chart_config(chart_config, columns)
            except json.JSONDecodeError:
                pass
            
            # Fallback: Sử dụng phân tích tự động
            return self._auto_select_chart_config(user_input, data_analysis, columns)
                
        except Exception as e:
            print(f" Lỗi phân tích yêu cầu biểu đồ: {e}")
            # Fallback đơn giản
            return {
                "chart_type": "bar",
                "x_column": data.get("columns", ["category"])[0],
                "y_column": data.get("columns", ["category", "value"])[1] if len(data.get("columns", [])) > 1 else "value",
                "title": "Biểu đồ",
                "reasoning": "Fallback mặc định"
            }
    
    def _clean_data_for_analysis(self, columns: List[str], data_rows: List[List]) -> str:
        """Làm sạch dữ liệu trước khi phân tích"""
        try:
            # Làm sạch tên cột
            cleaned_columns = []
            for col in columns:
                clean_col = str(col).strip()
                clean_col = re.sub(r'[^\w\s\-]', '', clean_col)
                if len(clean_col) > 20:
                    clean_col = clean_col[:17] + "..."
                cleaned_columns.append(clean_col)
            
            # Làm sạch dữ liệu mẫu
            cleaned_rows = []
            for row in data_rows[:3]:  # Chỉ lấy 3 dòng đầu
                cleaned_row = []
                for cell in row:
                    clean_cell = str(cell).strip()
                    if len(clean_cell) > 30:
                        clean_cell = clean_cell[:27] + "..."
                    cleaned_row.append(clean_cell)
                cleaned_rows.append(cleaned_row)
            
            return f"Cột: {cleaned_columns}\nDữ liệu mẫu: {cleaned_rows}"
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return f"Cột: {columns}\nDữ liệu mẫu: {data_rows[:3] if data_rows else []}"
    
    def _analyze_data_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Tự động phân tích cấu trúc dữ liệu để gợi ý trục X, Y"""
        try:
            columns = data.get("columns", [])
            data_rows = data.get("data", [])
            
            if not columns or not data_rows:
                return {
                    "available_columns": [],
                    "numeric_columns": [],
                    "categorical_columns": [],
                    "suggested_x": "",
                    "suggested_y": "",
                    "data_types": {}
                }
            
            # Phân tích từng cột
            numeric_columns = []
            categorical_columns = []
            data_types = {}
            
            for i, col in enumerate(columns):
                try:
                    # Lấy mẫu dữ liệu từ cột
                    sample_values = [row[i] for row in data_rows[:10] if i < len(row)]
                    
                    # Kiểm tra xem cột có phải số không
                    is_numeric = True
                    for val in sample_values:
                        try:
                            float(str(val).replace(',', '').replace('Decimal(', '').replace(')', ''))
                        except:
                            is_numeric = False
                            break
                    
                    if is_numeric:
                        numeric_columns.append(col)
                        data_types[col] = "numeric"
                    else:
                        categorical_columns.append(col)
                        data_types[col] = "categorical"
                        
                except Exception as e:
                    print(f"Error analyzing column {col}: {e}")
                    categorical_columns.append(col)
                    data_types[col] = "categorical"
            
            # Gợi ý trục X và Y
            suggested_x = ""
            suggested_y = ""
            
            # Trục X: Ưu tiên cột phân loại, nếu không có thì dùng cột đầu tiên
            if categorical_columns:
                suggested_x = categorical_columns[0]
            elif columns:
                suggested_x = columns[0]
            
            # Trục Y: Ưu tiên cột số, nếu không có thì dùng cột thứ 2
            if numeric_columns:
                suggested_y = numeric_columns[0]
            elif len(columns) > 1:
                suggested_y = columns[1]
            elif columns:
                suggested_y = columns[0]
            
            return {
                "available_columns": columns,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "suggested_x": suggested_x,
                "suggested_y": suggested_y,
                "data_types": data_types
            }
            
        except Exception as e:
            print(f"Error analyzing data structure: {e}")
            return {
                "available_columns": data.get("columns", []),
                "numeric_columns": [],
                "categorical_columns": data.get("columns", []),
                "suggested_x": data.get("columns", [""])[0],
                "suggested_y": data.get("columns", ["", ""])[1] if len(data.get("columns", [])) > 1 else data.get("columns", [""])[0],
                "data_types": {}
            }
    
    def _auto_select_chart_config(self, user_input: str, data_analysis: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Tự động chọn cấu hình biểu đồ dựa trên phân tích dữ liệu"""
        try:
            numeric_cols = data_analysis.get("numeric_columns", [])
            categorical_cols = data_analysis.get("categorical_columns", [])
            suggested_x = data_analysis.get("suggested_x", "")
            suggested_y = data_analysis.get("suggested_y", "")
            
            # Phân tích yêu cầu người dùng
            user_lower = user_input.lower()
            
            # Quyết định loại biểu đồ
            chart_type = "bar"  # Mặc định
            
            if "tròn" in user_lower or "pie" in user_lower or "phần trăm" in user_lower:
                chart_type = "pie"
            elif "đường" in user_lower or "line" in user_lower or "xu hướng" in user_lower:
                chart_type = "line"
            elif "phân tán" in user_lower or "scatter" in user_lower or "tương quan" in user_lower:
                chart_type = "scatter"
            elif "histogram" in user_lower or "phân phối" in user_lower:
                chart_type = "histogram"
            else:
                # Tự động chọn dựa trên dữ liệu
                if len(numeric_cols) >= 2:
                    chart_type = "scatter"
                elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                    chart_type = "bar"
                elif len(numeric_cols) >= 1:
                    chart_type = "histogram"
                else:
                    chart_type = "bar"
            
            # Quyết định trục X và Y
            x_column = suggested_x
            y_column = suggested_y
            
            # Điều chỉnh cho từng loại biểu đồ
            if chart_type == "pie":
                # Pie chart: X = category, Y = value
                if categorical_cols and numeric_cols:
                    x_column = categorical_cols[0]
                    y_column = numeric_cols[0]
                elif columns:
                    x_column = columns[0]
                    y_column = columns[1] if len(columns) > 1 else columns[0]
            elif chart_type == "scatter":
                # Scatter plot: cần 2 cột số
                if len(numeric_cols) >= 2:
                    x_column = numeric_cols[0]
                    y_column = numeric_cols[1]
                elif len(columns) >= 2:
                    x_column = columns[0]
                    y_column = columns[1]
            elif chart_type == "histogram":
                # Histogram: chỉ cần 1 cột số
                if numeric_cols:
                    x_column = numeric_cols[0]
                    y_column = numeric_cols[0]
                else:
                    x_column = columns[0] if columns else "value"
                    y_column = x_column
            
            # Tạo title
            title = self._generate_chart_title(user_input, chart_type, x_column, y_column)
            
            return {
                "chart_type": chart_type,
                "x_column": x_column,
                "y_column": y_column,
                "title": title,
                "reasoning": f"Tự động chọn {chart_type} chart với X={x_column}, Y={y_column} dựa trên phân tích dữ liệu"
            }
            
        except Exception as e:
            print(f"Error in auto select chart config: {e}")
            return {
                "chart_type": "bar",
                "x_column": columns[0] if columns else "category",
                "y_column": columns[1] if len(columns) > 1 else "value",
                "title": "Biểu đồ",
                "reasoning": "Fallback tự động"
            }
    
    def _generate_chart_title(self, user_input: str, chart_type: str, x_column: str, y_column: str) -> str:
        """Tự động tạo tiêu đề biểu đồ"""
        try:
            # Từ khóa trong yêu cầu
            if "lương" in user_input.lower():
                if "trung bình" in user_input.lower():
                    return f"So sánh mức lương trung bình giữa các {x_column}"
                else:
                    return f"Phân tích lương theo {x_column}"
            elif "nhân viên" in user_input.lower():
                return f"Thống kê nhân viên theo {x_column}"
            elif "phòng ban" in user_input.lower():
                return f"Phân tích theo phòng ban"
            else:
                # Tạo title dựa trên loại biểu đồ
                chart_names = {
                    "bar": "Biểu đồ cột",
                    "pie": "Biểu đồ tròn", 
                    "line": "Biểu đồ đường",
                    "scatter": "Biểu đồ phân tán",
                    "histogram": "Biểu đồ histogram"
                }
                return f"{chart_names.get(chart_type, 'Biểu đồ')} - {x_column} vs {y_column}"
                
        except Exception as e:
            print(f"Error generating title: {e}")
            return f"Biểu đồ {chart_type}"
    
    def _validate_chart_config(self, config: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Validate và làm sạch chart config"""
        try:
            # Validate chart_type
            valid_types = ["bar", "pie", "line", "scatter", "histogram"]
            if config.get("chart_type") not in valid_types:
                config["chart_type"] = "bar"
            
            # Validate columns
            if config.get("x_column") not in columns:
                config["x_column"] = columns[0] if columns else "category"
            
            if config.get("y_column") not in columns:
                config["y_column"] = columns[1] if len(columns) > 1 else "value"
            
            # Clean title
            title = config.get("title", "")
            if len(title) > 50:
                title = title[:47] + "..."
            config["title"] = title
            
            return config
        except Exception as e:
            print(f"Error validating config: {e}")
            return {
                "chart_type": "bar",
                "x_column": columns[0] if columns else "category",
                "y_column": columns[1] if len(columns) > 1 else "value",
                "title": "Biểu đồ",
                "reasoning": "Fallback sau validation"
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
            
            # Chuẩn bị dữ liệu với xử lý lỗi tốt hơn
            x_data = []
            y_data = []
            
            for row in data_rows:
                try:
                    # Xử lý dữ liệu X - làm sạch tên phòng ban
                    x_val = str(row[x_idx]).strip()
                    # Loại bỏ các ký tự đặc biệt nhưng giữ lại tiếng Việt
                    x_val = re.sub(r'[^\w\s\-àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', x_val)
                    # Giới hạn độ dài label nhưng không cắt quá ngắn
                    if len(x_val) > 25:
                        x_val = x_val[:22] + "..."
                    x_data.append(x_val)
                    
                    # Xử lý dữ liệu Y - cải thiện xử lý số
                    y_val = row[y_idx]
                    if isinstance(y_val, (int, float)):
                        y_data.append(float(y_val))
                    else:
                        # Thử chuyển đổi string thành số với xử lý Decimal
                        try:
                            val_str = str(y_val).replace(',', '').replace('Decimal(', '').replace(')', '')
                            y_data.append(float(val_str))
                        except:
                            # Nếu không thể chuyển đổi, thử tìm số trong string
                            numbers = re.findall(r'\d+\.?\d*', str(y_val))
                            if numbers:
                                y_data.append(float(numbers[0]))
                            else:
                                y_data.append(0)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            if not x_data or not y_data:
                return {"error": "Không có dữ liệu hợp lệ để vẽ biểu đồ"}
            
            # Tạo biểu đồ
            plt.figure(figsize=(12, 8))
            bars = plt.bar(x_data, y_data, color='skyblue', edgecolor='navy', alpha=0.7)
            
            # Thêm giá trị trên mỗi cột
            max_val = max(y_data) if y_data else 0
            for bar, value in zip(bars, y_data):
                if max_val > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.01, 
                            f'{value:.1f}', ha='center', va='bottom', fontsize=10)
            
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Danh mục', fontsize=12)
            plt.ylabel('Giá trị', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Lưu file
            chart_file = f"chart_{hash(user_input) % 1000000}.png"
            chart_path = self.chart_dir / chart_file
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {
                "chart_file": str(chart_path),
                "chart_type": "bar",
                "title": title,
                "data_points": len(data_rows),
                "x_labels": x_data,
                "y_values": y_data
            }
            
        except Exception as e:
            return {"error": f"Lỗi tạo biểu đồ cột: {str(e)}"}
    
    def _create_pie_chart(self, data_rows: List[List], x_idx: int, y_idx: int, title: str, user_input: str) -> Dict[str, Any]:
        """Tạo biểu đồ tròn"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # Chuẩn bị dữ liệu với xử lý lỗi tốt hơn
            labels = []
            values = []
            
            for row in data_rows:
                try:
                    # Xử lý labels - làm sạch tên phòng ban
                    label = str(row[x_idx]).strip()
                    # Loại bỏ các ký tự đặc biệt nhưng giữ lại tiếng Việt
                    label = re.sub(r'[^\w\s\-àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', label)
                    # Giới hạn độ dài label
                    if len(label) > 18:
                        label = label[:15] + "..."
                    labels.append(label)
                    
                    # Xử lý values - cải thiện xử lý số
                    val = row[y_idx]
                    if isinstance(val, (int, float)):
                        values.append(float(val))
                    else:
                        # Thử chuyển đổi string thành số với xử lý Decimal
                        try:
                            val_str = str(val).replace(',', '').replace('Decimal(', '').replace(')', '')
                            values.append(float(val_str))
                        except:
                            # Nếu không thể chuyển đổi, thử tìm số trong string
                            numbers = re.findall(r'\d+\.?\d*', str(val))
                            if numbers:
                                values.append(float(numbers[0]))
                            else:
                                values.append(0)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            if not labels or not values:
                return {"error": "Không có dữ liệu hợp lệ để vẽ biểu đồ"}
            
            # Loại bỏ các giá trị 0 hoặc âm
            filtered_data = [(label, val) for label, val in zip(labels, values) if val > 0]
            if not filtered_data:
                return {"error": "Tất cả giá trị đều bằng 0 hoặc âm"}
            
            labels, values = zip(*filtered_data)
            
            # Tạo biểu đồ
            plt.figure(figsize=(12, 10))
            colors = plt.cm.Set3(range(len(labels)))
            wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', 
                                             colors=colors, startangle=90, textprops={'fontsize': 10})
            
            # Tùy chỉnh text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            # Tùy chỉnh labels
            for text in texts:
                text.set_fontsize(10)
                text.set_fontweight('bold')
            
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.axis('equal')
            
            # Lưu file
            chart_file = f"chart_{hash(user_input) % 1000000}.png"
            chart_path = self.chart_dir / chart_file
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {
                "chart_file": str(chart_path),
                "chart_type": "pie",
                "title": title,
                "data_points": len(data_rows),
                "labels": list(labels),
                "values": list(values)
            }
            
        except Exception as e:
            return {"error": f"Lỗi tạo biểu đồ tròn: {str(e)}"}
    
    def _create_donut_chart(self, suitable_percent: float, unsuitable_percent: float, title: str = "Đánh Giá Phù Hợp CV") -> Dict[str, Any]:
        """Tạo biểu đồ donut cho kết quả CV"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # Dữ liệu
            labels = ['Phù hợp', 'Không phù hợp']
            sizes = [suitable_percent, unsuitable_percent]
            colors = ['#ff4444', '#44ff44']  # Đỏ cho phù hợp, xanh cho không phù hợp
            explode = (0.05, 0.05)  # Tách nhẹ các phần
            
            # Tạo figure - rất nhỏ để vừa trong chat
            fig, ax = plt.subplots(figsize=(3, 2.5))
            
            # Tạo donut chart
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                pctdistance=0.85,
                textprops={'fontsize': 6, 'weight': 'bold'}
            )
            
            # Tạo lỗ ở giữa để tạo donut
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            ax.add_artist(centre_circle)
            
            # Thêm tổng điểm ở giữa
            total_score = suitable_percent
            ax.text(0, 0, f'{total_score}%\nTỔNG ĐIỂM', 
                   ha='center', va='center', fontsize=7, fontweight='bold')
            
            # Cấu hình
            ax.set_title(title, fontsize=8, fontweight='bold', pad=5)
            
            # Lưu biểu đồ
            chart_filename = f"donut_chart_{int(suitable_percent)}_{int(unsuitable_percent)}.png"
            chart_path = self.chart_dir / chart_filename
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {
                "chart_type": "donut",
                "chart_file": str(chart_path),
                "chart_filename": chart_filename,
                "title": title,
                "data": {
                    "suitable_percent": suitable_percent,
                    "unsuitable_percent": unsuitable_percent,
                    "total_score": total_score
                }
            }
            
        except Exception as e:
            return {"error": f"Lỗi tạo biểu đồ donut: {str(e)}"}
    
    def _create_line_chart(self, data_rows: List[List], x_idx: int, y_idx: int, title: str, user_input: str) -> Dict[str, Any]:
        """Tạo biểu đồ đường"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # Chuẩn bị dữ liệu với xử lý lỗi tốt hơn
            x_data = []
            y_data = []
            
            for row in data_rows:
                try:
                    # Xử lý dữ liệu X - làm sạch tên phòng ban
                    x_val = str(row[x_idx]).strip()
                    # Loại bỏ các ký tự đặc biệt nhưng giữ lại tiếng Việt
                    x_val = re.sub(r'[^\w\s\-àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', x_val)
                    # Giới hạn độ dài label
                    if len(x_val) > 20:
                        x_val = x_val[:17] + "..."
                    x_data.append(x_val)
                    
                    # Xử lý dữ liệu Y - cải thiện xử lý số
                    y_val = row[y_idx]
                    if isinstance(y_val, (int, float)):
                        y_data.append(float(y_val))
                    else:
                        # Thử chuyển đổi string thành số với xử lý Decimal
                        try:
                            val_str = str(y_val).replace(',', '').replace('Decimal(', '').replace(')', '')
                            y_data.append(float(val_str))
                        except:
                            # Nếu không thể chuyển đổi, thử tìm số trong string
                            numbers = re.findall(r'\d+\.?\d*', str(y_val))
                            if numbers:
                                y_data.append(float(numbers[0]))
                            else:
                                y_data.append(0)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            if not x_data or not y_data:
                return {"error": "Không có dữ liệu hợp lệ để vẽ biểu đồ"}
            
            # Tạo biểu đồ
            plt.figure(figsize=(12, 8))
            plt.plot(x_data, y_data, marker='o', linewidth=3, markersize=8, color='blue', alpha=0.8)
            
            # Thêm điểm dữ liệu với cải thiện
            for i, (x, y) in enumerate(zip(x_data, y_data)):
                if i % max(1, len(x_data) // 10) == 0:  # Chỉ hiển thị một số label để tránh chồng chéo
                    plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                                xytext=(0,15), ha='center', fontsize=9, 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Danh mục', fontsize=12)
            plt.ylabel('Giá trị', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Lưu file
            chart_file = f"chart_{hash(user_input) % 1000000}.png"
            chart_path = self.chart_dir / chart_file
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {
                "chart_file": str(chart_path),
                "chart_type": "line",
                "title": title,
                "data_points": len(data_rows),
                "x_labels": x_data,
                "y_values": y_data
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
        try:
            if isinstance(data, dict) and "columns" in data and "data" in data:
                # Kiểm tra nếu dữ liệu có cấu trúc đặc biệt từ Query Agent
                if data.get("columns") == ["result"] and len(data.get("data", [])) == 1:
                    # Dữ liệu có dạng: {'columns': ['result'], 'data': [["JSON_STRING"]]}
                    result_string = data["data"][0][0] if data["data"] and data["data"][0] else ""
                    if isinstance(result_string, str) and result_string.startswith("["):
                        print(f"Chart Agent: Detected special Query Agent format, parsing JSON string")
                        # Parse JSON string từ result
                        try:
                            # Clean up the JSON string
                            json_str = result_string.strip()
                            print(f"Chart Agent: Raw JSON string: {json_str[:200]}...")
                            
                            # Replace Decimal('...') with just the number
                            json_str = re.sub(r"Decimal\('([^']+)'\)", r'\1', json_str)
                            
                            # Convert single quotes to double quotes for valid JSON
                            json_str = json_str.replace("'", '"')
                            print(f"Chart Agent: Cleaned JSON string: {json_str[:200]}...")
                            
                            json_data = json.loads(json_str)
                            print(f"Chart Agent: Parsed JSON data: {json_data}")
                            
                            if isinstance(json_data, list) and json_data:
                                # Convert to DataFrame and then to our format
                                df = pd.DataFrame(json_data)
                                print(f"Chart Agent: Successfully parsed Query Agent JSON with {len(df)} rows and {len(df.columns)} columns")
                                print(f"Chart Agent: DataFrame columns: {df.columns.tolist()}")
                                print(f"Chart Agent: DataFrame data: {df.values.tolist()}")
                                return {
                                    "columns": df.columns.tolist(),
                                    "data": df.values.tolist()
                                }
                        except Exception as e:
                            print(f"Error parsing Query Agent JSON: {e}")
                            # Try alternative parsing with ast.literal_eval
                            try:
                                # Convert back to single quotes for ast.literal_eval
                                eval_str = result_string.strip()
                                # Replace Decimal('...') with just the number
                                eval_str = re.sub(r"Decimal\('([^']+)'\)", r'\1', eval_str)
                                json_data = ast.literal_eval(eval_str)
                                print(f"Chart Agent: Successfully parsed with ast.literal_eval: {json_data}")
                                
                                if isinstance(json_data, list) and json_data:
                                    df = pd.DataFrame(json_data)
                                    print(f"Chart Agent: Successfully parsed Query Agent JSON with {len(df)} rows and {len(df.columns)} columns")
                                    return {
                                        "columns": df.columns.tolist(),
                                        "data": df.values.tolist()
                                    }
                            except Exception as e2:
                                print(f"Error with ast.literal_eval: {e2}")
                
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
                    print(f"Chart Agent: Processing string data of length {len(data)}")
                    # First, try to parse JSON data that might be embedded in the string
                    # Look for patterns like: [{'ten_phong_ban': 'Phòng Nhân sự', 'luong_trung_binh': Decimal('12888888.888888888889')}, ...]
                    json_pattern = r'\[.*?\]'
                    json_matches = re.findall(json_pattern, data, re.DOTALL)
                    print(f"Chart Agent: Found {len(json_matches)} JSON matches")
                    
                    for json_str in json_matches:
                        try:
                            # Clean up the JSON string
                            json_str = json_str.strip()
                            print(f"Chart Agent: Raw JSON string: {json_str[:200]}...")
                            
                            # Replace Decimal('...') with just the number
                            json_str = re.sub(r"Decimal\('([^']+)'\)", r'\1', json_str)
                            print(f"Chart Agent: Cleaned JSON string: {json_str[:200]}...")
                            
                            json_data = json.loads(json_str)
                            print(f"Chart Agent: Parsed JSON data: {json_data}")
                            
                            if isinstance(json_data, list) and json_data:
                                # Convert to DataFrame and then to our format
                                df = pd.DataFrame(json_data)
                                print(f"Chart Agent: Successfully parsed JSON data with {len(df)} rows and {len(df.columns)} columns")
                                print(f"Chart Agent: DataFrame columns: {df.columns.tolist()}")
                                print(f"Chart Agent: DataFrame data: {df.values.tolist()}")
                                return {
                                    "columns": df.columns.tolist(),
                                    "data": df.values.tolist()
                                }
                        except Exception as e:
                            print(f"Error parsing JSON: {e}")
                            continue
                    
                    # Try to extract data from markdown table or text
                    
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
                    
                    # Handle complex data structures that might be causing the X-axis issue
                    # Look for patterns like tuples or complex objects
                    complex_pattern = r'\(([^,]+),\s*([^)]+)\)'
                    complex_matches = re.findall(complex_pattern, data)
                    if complex_matches:
                        columns = ["Category", "Value"]
                        data_rows = []
                        for match in complex_matches:
                            try:
                                # Clean up the values
                                category = match[0].strip().strip('"\'')
                                value = match[1].strip().strip('"\'')
                                # Try to convert to number
                                try:
                                    value_num = float(value)
                                except:
                                    value_num = 0
                                data_rows.append([category, value_num])
                            except:
                                continue
                        
                        if data_rows:
                            return {
                                "columns": columns,
                                "data": data_rows
                            }
                    
                except Exception as e:
                    print(f"Error parsing string data: {e}")
            
            # Handle tuple data that might be causing issues
            if isinstance(data, (tuple, list)) and data:
                try:
                    if isinstance(data[0], (tuple, list)):
                        # List of tuples/lists
                        columns = ["Category", "Value"]
                        data_rows = []
                        for item in data:
                            if len(item) >= 2:
                                try:
                                    category = str(item[0]).strip()
                                    value = float(item[1]) if isinstance(item[1], (int, float)) else 0
                                    data_rows.append([category, value])
                                except:
                                    continue
                        if data_rows:
                            return {
                                "columns": columns,
                                "data": data_rows
                            }
                except:
                    pass
            
        except Exception as e:
            print(f"Error in _normalize_data: {e}")
        
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
        Xử lý yêu cầu tạo biểu đồ với khả năng tự động phân tích dữ liệu
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
                        "usage": "Cung cấp dữ liệu để tạo biểu đồ",
                        "auto_analysis": "Tự động phân tích dữ liệu và chọn trục X, Y phù hợp"
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
            print(f"Normalized columns: {normalized_data.get('columns', [])}")
            if normalized_data.get('data'):
                print(f"Sample data: {normalized_data.get('data', [])[:2]}")
            
            if not normalized_data.get("data"):
                return {
                    "agent": "chart_agent",
                    "status": "error",
                    "error": f"Không thể parse dữ liệu từ Query Agent. Data type: {type(data)}"
                }
            
            # Tự động phân tích dữ liệu và chọn loại biểu đồ phù hợp
            print(f"Chart Agent: Tự động phân tích dữ liệu và chọn loại biểu đồ...")
            chart_config = await self._analyze_chart_requirements(user_input, normalized_data)
            
            print(f"Chart Agent: Đã chọn biểu đồ {chart_config['chart_type']}")
            print(f"Chart Agent: Trục X = {chart_config['x_column']}, Trục Y = {chart_config['y_column']}")
            print(f"Chart Agent: Lý do: {chart_config['reasoning']}")
            
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
                    "data_summary": f"Đã tạo biểu đồ {chart_config['chart_type']} với {len(normalized_data.get('data', []))} điểm dữ liệu",
                    "auto_analysis": {
                        "detected_columns": normalized_data.get('columns', []),
                        "chart_type_selected": chart_config['chart_type'],
                        "x_axis": chart_config['x_column'],
                        "y_axis": chart_config['y_column'],
                        "reasoning": chart_config['reasoning']
                    }
                }
            }
            
        except Exception as e:
            return {
                "agent": "chart_agent",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def create_smart_chart(self, user_input: str, data: Any) -> Dict[str, Any]:
        """
        Tạo biểu đồ thông minh - tự động phân tích dữ liệu và chọn loại biểu đồ phù hợp nhất
        """
        try:
            print(f"Chart Agent: Tạo biểu đồ thông minh cho '{user_input}'")
            
            # Chuẩn hóa dữ liệu
            normalized_data = self._normalize_data(data)
            
            if not normalized_data.get("data"):
                return {
                    "agent": "chart_agent",
                    "status": "error",
                    "error": "Không có dữ liệu để tạo biểu đồ"
                }
            
            # Phân tích dữ liệu tự động
            data_analysis = self._analyze_data_structure(normalized_data)
            print(f"Chart Agent: Phân tích dữ liệu - {len(data_analysis.get('numeric_columns', []))} cột số, {len(data_analysis.get('categorical_columns', []))} cột phân loại")
            
            # Tự động chọn cấu hình biểu đồ
            chart_config = self._auto_select_chart_config(user_input, data_analysis, normalized_data.get("columns", []))
            print(f"Chart Agent: Đã chọn {chart_config['chart_type']} với X={chart_config['x_column']}, Y={chart_config['y_column']}")
            
            # Tạo biểu đồ
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
                    "smart_analysis": {
                        "data_structure": data_analysis,
                        "chart_config": chart_config,
                        "auto_selection": "Tự động chọn loại biểu đồ và trục dựa trên cấu trúc dữ liệu"
                    }
                }
            }
            
        except Exception as e:
            return {
                "agent": "chart_agent",
                "status": "error",
                "error": str(e)
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
