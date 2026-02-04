import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

########## ------ Đọc và làm sạch dữ liệu ------- ##########
# Đọc file CSV
df = pd.read_csv("../datasets/hurdat2.csv")

# Chuyển latitude, longitude sang dạng số
def parse_lat(lat):
    if pd.isna(lat): return np.nan
    return float(lat[:-1]) * (1 if lat[-1] == "N" else -1)

def parse_lon(lon):
    if pd.isna(lon): return np.nan
    return float(lon[:-1]) * (-1 if lon[-1] == "W" else 1)

df["lat"] = df["latitude"].apply(parse_lat)
df["lon"] = df["longitude"].apply(parse_lon)

# Chuyển date/time sang datetime
df["datetime"] = pd.to_datetime(df["date"].astype(str) + df["time"].astype(str).str.zfill(4), errors="coerce")

# Thay thế -999 bằng NaN
df = df.replace(-999, np.nan)

# ########## ------ Phân bố tốc độ gió tối đa ------- ##########

# plt.figure(figsize=(8,5))
# sns.histplot(df["maximum_sustained_wind_knots"].dropna(), bins=30, kde=True)
# plt.title("Phân bố tốc độ gió tối đa (knots)")
# plt.xlabel("Tốc độ gió (knots)")
# plt.ylabel("Số lượng quan sát")
# plt.show()

# ########## ------ Trung bình gió theo năm (xu hướng dài hạn) ------- ##########

# df["year"] = df["datetime"].dt.year
# yearly = df.groupby("year")["maximum_sustained_wind_knots"].mean().dropna()

# plt.figure(figsize=(10,5))
# plt.plot(yearly.index, yearly.values)
# plt.title("Xu hướng trung bình gió tối đa theo năm")
# plt.xlabel("Năm")
# plt.ylabel("Gió tối đa trung bình (knots)")
# plt.show()

# ########## ------ Heatmap tần suất bão theo vĩ độ và kinh độ ------- ##########

# plt.figure(figsize=(10,6))
# sns.kdeplot(
#     x=df["lon"], y=df["lat"],
#     fill=True, cmap="Reds", thresh=0.05
# )
# plt.title("Heatmap tần suất xuất hiện bão (Density)")
# plt.xlabel("Kinh độ")
# plt.ylabel("Vĩ độ")
# plt.show()

########## ------ Top 10 cơn bão có tốc độ gió tối đa cao nhất ------- ##########

top_storms = df.groupby("storm_id")["maximum_sustained_wind_knots"].max().sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top_storms.values, y=top_storms.index, palette="Reds_r")
plt.title("Top 10 cơn bão có tốc độ gió tối đa cao nhất")
plt.xlabel("Tốc độ gió tối đa (knots)")
plt.ylabel("Mã bão (storm_id)")
plt.show()

########## ------ Số lượng bão theo tháng trong năm ------- ##########

df["month"] = df["datetime"].dt.month
monthly_counts = df.groupby("month")["storm_id"].nunique()

plt.figure(figsize=(8,5))
sns.barplot(x=monthly_counts.index, y=monthly_counts.values, color="skyblue")
plt.title("Số lượng bão theo tháng trong năm")
plt.xlabel("Tháng")
plt.ylabel("Số lượng bão")
plt.show()

########## ------ Mối quan hệ giữa áp suất tâm bão và tốc độ gió ------- ##########

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df, 
    x="central_pressure_mb", 
    y="maximum_sustained_wind_knots", 
    alpha=0.6
)
plt.title("Quan hệ giữa áp suất tâm bão và tốc độ gió")
plt.xlabel("Áp suất trung tâm (mb)")
plt.ylabel("Gió tối đa (knots)")
plt.show()

############ ------ Phân bố vị trí theo cấp độ bão (Status) ------- ##########

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="status_of_system", order=df["status_of_system"].value_counts().index)
plt.title("Phân bố trạng thái hệ thống (HU, TS, EX, ...)")
plt.xlabel("Trạng thái")
plt.ylabel("Số lượng quan sát")
plt.show()

########### ------ Chiều dài “vòng đời” của từng bão ------- ##########

storm_duration = df.groupby("storm_id")["datetime"].count().sort_values(ascending=False).head(15)

plt.figure(figsize=(9,5))
sns.barplot(x=storm_duration.values, y=storm_duration.index, color="lightgreen")
plt.title("Thời gian tồn tại (số bản ghi) của Top 15 cơn bão")
plt.xlabel("Số bản ghi (mỗi 6h)")
plt.ylabel("Mã bão (storm_id)")
plt.show()

