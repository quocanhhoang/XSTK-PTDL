import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

df = pd.read_excel('D:\Python\Xác suất thống kê và phân tích dữ liệu\BaiTapLon\Dự liệu sử dụng người dùng trên website.xlsx')
print(df.head(250))

# Bước 2: Làm sạch dữ liệu
# Đặt lại tên cột cho dễ dùng
df.columns = ["ma_nguoi_dung", "thoi_gian_o_lai", "so_trang_xem", "ty_le_quay_lai"]

# Kiểm tra giá trị thiếu
print("Số lượng giá trị thiếu:\n", df.isnull().sum())

# Chuẩn hóa kiểu dữ liệu về dạng số
df["thoi_gian_o_lai"] = pd.to_numeric(df["thoi_gian_o_lai"], errors="coerce")
df["so_trang_xem"] = pd.to_numeric(df["so_trang_xem"], errors="coerce")
df["ty_le_quay_lai"] = pd.to_numeric(df["ty_le_quay_lai"], errors="coerce")

# In ra dữ liệu sau khi đã làm sạch
print("\nDữ liệu sau khi làm sạch:")
print(df.head(250))
# Xuất dữ liệu đã làm sạch ra file CSV để kiểm tra nếu cần và tránh bị lag do quá nhiều dòng khi in ra console
df.to_csv("Dữ liệu đã làm sạch.csv", index=False)

# Bước 3: Phân tích khám phá dữ liệu (EDA)
print("Thống kê mô tả cơ bản")
print(df.describe())

# In thêm các chỉ số riêng lẻ
print("\nChỉ số chi tiết:")
for col in ["thoi_gian_o_lai", "so_trang_xem", "ty_le_quay_lai"]:
    print(f"\nBiến: {col}")
    print("Trung bình:", df[col].mean())
    print("Trung vị:", df[col].median())
    print("Phương sai:", df[col].var())
    print("Giá trị nhỏ nhất:", df[col].min())
    print("Giá trị lớn nhất:", df[col].max())

# Biểu đồ phân phối thời gian ở lại
plt.figure(figsize=(6,4))
sns.histplot(df["thoi_gian_o_lai"], bins=20, kde=True)
plt.title("Phân phối thời gian ở lại (giây)")
plt.show()

# Biểu đồ phân phối số trang xem
plt.figure(figsize=(6,4))
sns.histplot(df["so_trang_xem"], bins=20, kde=False)
plt.title("Phân phối số trang xem")
plt.show()

# Biểu đồ tỷ lệ quay lại
plt.figure(figsize=(6,4))
sns.histplot(df["ty_le_quay_lai"], bins=10, kde=False)
plt.title("Phân phối tỷ lệ quay lại (biểu đồ cột)")
plt.xlabel("Tỷ lệ quay lại")
plt.ylabel("Số lượng người dùng")
plt.show()

# Bước 4. Xây dựng mô hình
y = df["ty_le_quay_lai"]

# -------------------------
# 1. Hồi quy tuyến tính: dự đoán từ thời gian ở lại
X1 = df["thoi_gian_o_lai"].values.reshape(-1,1)
model1 = LinearRegression()
model1.fit(X1, y)
y_pred1 = model1.predict(X1)

beta_0 = model1.intercept_
beta_1 = model1.coef_[0]
r2 = r2_score(y, y_pred1)
mse = mean_squared_error(y, y_pred1)

print("Hồi quy tuyến tính: thời gian ở lại và tỷ lệ quay lại")
print(f"Phương trình hồi quy: y = {beta_0:.2f} + {beta_1:.2f}x")
print(f"Hệ số chặn (beta_0): {beta_0}")
print(f"Hệ số dốc (beta_1): {beta_1}")
print(f"Hệ số xác định (R^2): {r2}")
print(f"Sai số bình phương trung bình (MSE): {mse}")

plt.scatter(X1, y, color="blue", label="Thực tế")
plt.plot(X1, y_pred1, color="red", label="Hồi quy")
plt.title("Hồi quy tuyến tính: Thời gian ở lại và Tỷ lệ quay lại")
plt.xlabel("Thời gian ở lại (giây)")
plt.ylabel("Tỷ lệ quay lại")
plt.legend()
plt.show()

# -------------------------
# 2. Hồi quy tuyến tính: dự đoán từ số trang xem
X2 = df["so_trang_xem"].values.reshape(-1,1)
model2 = LinearRegression()
model2.fit(X2, y)
y_pred2 = model2.predict(X2)

beta_0 = model2.intercept_
beta_1 = model2.coef_[0]
r2 = r2_score(y, y_pred2)
mse = mean_squared_error(y, y_pred2)

print("Hồi quy tuyến tính: số trang xem và tỷ lệ quay lại")
print(f"Phương trình hồi quy: y = {beta_0:.2f} + {beta_1:.2f}x")
print(f"Hệ số chặn (beta_0): {beta_0}")
print(f"Hệ số dốc (beta_1): {beta_1}")
print(f"Hệ số xác định (R^2): {r2}")
print(f"Sai số bình phương trung bình (MSE): {mse}")

plt.scatter(X2, y, color="green", label="Thực tế")
plt.plot(X2, y_pred2, color="red", label="Hồi quy")
plt.title("Hồi quy tuyến tính: Số trang xem và Tỷ lệ quay lại")
plt.xlabel("Số trang xem")
plt.ylabel("Tỷ lệ quay lại")
plt.legend()
plt.show()