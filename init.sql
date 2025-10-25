-- Initialize HR Database with sample data
CREATE TABLE IF NOT EXISTS employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(50) NOT NULL,
    position VARCHAR(50) NOT NULL,
    salary DECIMAL(10,2),
    hire_date DATE,
    gender VARCHAR(10),
    education VARCHAR(50)
);

-- Insert sample data
INSERT INTO employees (name, department, position, salary, hire_date, gender, education) VALUES
('Nguyễn Văn An', 'IT', 'Developer', 15000000, '2023-01-15', 'Nam', 'Đại học'),
('Trần Thị Bình', 'HR', 'HR Manager', 18000000, '2022-03-20', 'Nữ', 'Thạc sĩ'),
('Lê Văn Cường', 'IT', 'Senior Developer', 20000000, '2021-06-10', 'Nam', 'Đại học'),
('Phạm Thị Dung', 'Marketing', 'Marketing Specialist', 12000000, '2023-02-28', 'Nữ', 'Đại học'),
('Hoàng Văn Em', 'Sales', 'Sales Manager', 16000000, '2022-11-15', 'Nam', 'Đại học'),
('Vũ Thị Phương', 'HR', 'HR Specialist', 13000000, '2023-04-01', 'Nữ', 'Đại học'),
('Đặng Văn Giang', 'IT', 'DevOps Engineer', 19000000, '2022-08-20', 'Nam', 'Thạc sĩ'),
('Bùi Thị Hoa', 'Finance', 'Accountant', 11000000, '2023-01-10', 'Nữ', 'Đại học'),
('Ngô Văn Ích', 'Sales', 'Sales Executive', 14000000, '2022-12-05', 'Nam', 'Cao đẳng'),
('Lý Thị Kim', 'Marketing', 'Content Creator', 10000000, '2023-03-15', 'Nữ', 'Đại học'),
('Võ Văn Long', 'IT', 'Frontend Developer', 13500000, '2023-02-01', 'Nam', 'Đại học'),
('Đinh Thị Mai', 'HR', 'Recruiter', 12000000, '2022-09-10', 'Nữ', 'Đại học'),
('Phan Văn Nam', 'Sales', 'Sales Representative', 12500000, '2023-01-20', 'Nam', 'Cao đẳng'),
('Trương Thị Oanh', 'Finance', 'Financial Analyst', 15000000, '2022-07-15', 'Nữ', 'Thạc sĩ'),
('Lưu Văn Phúc', 'IT', 'Backend Developer', 17000000, '2022-05-30', 'Nam', 'Đại học'),
('Nguyễn Thị Quỳnh', 'Marketing', 'Social Media Manager', 11500000, '2023-02-14', 'Nữ', 'Đại học'),
('Trần Văn Rồng', 'Sales', 'Sales Director', 22000000, '2021-04-01', 'Nam', 'Thạc sĩ'),
('Lê Thị Sương', 'HR', 'HR Director', 25000000, '2021-01-15', 'Nữ', 'Thạc sĩ'),
('Phạm Văn Tài', 'IT', 'Tech Lead', 28000000, '2020-08-20', 'Nam', 'Thạc sĩ'),
('Hoàng Thị Uyên', 'Finance', 'Finance Manager', 20000000, '2022-03-01', 'Nữ', 'Thạc sĩ'),
('Vũ Văn Việt', 'Marketing', 'Marketing Manager', 18000000, '2022-01-10', 'Nam', 'Đại học'),
('Đặng Thị Xuân', 'Sales', 'Sales Coordinator', 13000000, '2023-03-01', 'Nữ', 'Đại học'),
('Bùi Văn Yên', 'IT', 'QA Engineer', 14000000, '2022-10-15', 'Nam', 'Đại học'),
('Ngô Thị Zara', 'Finance', 'Budget Analyst', 16000000, '2022-06-20', 'Nữ', 'Thạc sĩ'),
('Lý Văn Anh', 'Marketing', 'Brand Manager', 19000000, '2021-12-01', 'Nam', 'Thạc sĩ');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department);
CREATE INDEX IF NOT EXISTS idx_employees_position ON employees(position);
CREATE INDEX IF NOT EXISTS idx_employees_gender ON employees(gender);
CREATE INDEX IF NOT EXISTS idx_employees_education ON employees(education);
