import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Đọc file CSV
file_path = "../datasets/LD2011_2014.csv"  # Thay bằng đường dẫn thực tế
df = pd.read_csv(file_path)

# Lọc các cột số
numeric_columns = df.select_dtypes(include=['number']).columns

# Tạo thư mục lưu ảnh
output_dir = "distribution_plots"
os.makedirs(output_dir, exist_ok=True)

# Vẽ và lưu từng biểu đồ phân phối
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, bins=50, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

    # Lưu ảnh
    output_path = os.path.join(output_dir, f"{col}.png")
    plt.savefig(output_path)
    plt.close()

print(f"Đã lưu {len(numeric_columns)} biểu đồ vào thư mục '{output_dir}'")
