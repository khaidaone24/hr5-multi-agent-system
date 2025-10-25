import os
import psycopg2
from sqlalchemy import create_engine, text
import random
from datetime import datetime, timedelta

# Lấy DB_LINK từ biến môi trường
DB_LINK = os.getenv("DB_LINK")

if not DB_LINK:
    print("🚨 Lỗi: Biến môi trường DB_LINK không được tìm thấy.")
    print("Vui lòng cấu hình DB_LINK trong Railway Variables hoặc file .env.")
    exit(1)

print(f"🚀 Đang kết nối đến database: {DB_LINK.split('@')[-1]}")

try:
    # Sử dụng SQLAlchemy để kết nối
    engine = create_engine(DB_LINK)

    with engine.connect() as connection:
        # Tạo bảng nhan_vien nếu chưa có
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS nhan_vien (
            id SERIAL PRIMARY KEY,
            ten VARCHAR(100) NOT NULL,
            phong_ban VARCHAR(50) NOT NULL,
            chuc_vu VARCHAR(50) NOT NULL,
            luong DECIMAL(10,2),
            ngay_vao_lam DATE,
            gioi_tinh VARCHAR(10),
            trinh_do VARCHAR(50)
        );
        """
        connection.execute(text(create_table_sql))
        print("✅ Đã kiểm tra/tạo bảng 'nhan_vien'.")

        # Xóa dữ liệu cũ nếu muốn chạy lại
        # connection.execute(text("DELETE FROM nhan_vien;"))
        # print("✅ Đã xóa dữ liệu cũ trong bảng 'nhan_vien'.")

        # Insert 20 nhân viên mẫu
        departments = ['IT', 'HR', 'Marketing', 'Sales', 'Finance']
        positions = ['Developer', 'Manager', 'Specialist', 'Analyst', 'Intern']
        genders = ['Nam', 'Nữ']
        educations = ['Đại học', 'Thạc sĩ', 'Cao đẳng']

        for i in range(1, 21):
            name = f"Nhân Viên {i}"
            department = random.choice(departments)
            position = random.choice(positions)
            salary = round(random.uniform(10000000, 30000000), 2)
            hire_date = datetime.now() - timedelta(days=random.randint(30, 1000))
            gender = random.choice(genders)
            education = random.choice(educations)

            insert_sql = text("""
            INSERT INTO nhan_vien (ten, phong_ban, chuc_vu, luong, ngay_vao_lam, gioi_tinh, trinh_do)
            VALUES (:ten, :phong_ban, :chuc_vu, :luong, :ngay_vao_lam, :gioi_tinh, :trinh_do);
            """)
            connection.execute(insert_sql, {
                'ten': name,
                'phong_ban': department,
                'chuc_vu': position,
                'luong': salary,
                'ngay_vao_lam': hire_date,
                'gioi_tinh': gender,
                'trinh_do': education
            })
        connection.commit()
        print("✅ Đã thêm 20 nhân viên mẫu vào bảng 'nhan_vien'.")

except Exception as e:
    print(f"🚨 Lỗi khi điền dữ liệu: {e}")
    exit(1)

print("🎉 Hoàn tất điền dữ liệu vào database!")
