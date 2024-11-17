import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Đọc tệp dữ liệu
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    columns_to_drop = [
        'CLIENTNUM', 
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    data_cleaned = data.drop(columns=columns_to_drop)
    data_cleaned['Attrition_Flag'] = LabelEncoder().fit_transform(data_cleaned['Attrition_Flag'])
    data_encoded = pd.get_dummies(data_cleaned, drop_first=True)
    return data_encoded

# Tải dữ liệu và chia tập
st.title("Dự đoán khả năng rời bỏ khách hàng")

# Đường dẫn mặc định
file_path = "BankChurners.csv"

# Tải dữ liệu
data = load_data(file_path)
# st.write("Dữ liệu ban đầu:", data.head())

X = data.drop(columns=['Attrition_Flag'])
y = data['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Dự đoán và đánh giá
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]

report = classification_report(y_test, predictions, target_names=['Existing Customer', 'Attrited Customer'])
auc_score = roc_auc_score(y_test, probabilities)

# st.subheader("Kết quả đánh giá mô hình")
# st.text(report)
# st.write(f"AUC: {auc_score:.4f}")

# # Thêm tính năng dự đoán cho khách hàng mới
# st.subheader("Dự đoán cho khách hàng mới")

# Giao diện Streamlit
st.title("Dự đoán khả năng rời bỏ khách hàng")

# Giao diện nhập liệu
st.sidebar.header("Thông tin khách hàng")
sample = []

# Tuổi khách hàng
age = st.sidebar.number_input("Tuổi khách hàng", min_value=18, max_value=100, value=30)
sample.append(age)

# Giới tính
gender = st.sidebar.selectbox("Giới tính", ["Nam", "Nữ", "Khác"])
# Mã hóa giới tính (giả định: Nam=0, Nữ=1, Khác=2)
gender_encoded = {"Nam": 0, "Nữ": 1, "Khác": 2}
sample.append(gender_encoded[gender])

# Trình độ học vấn
education_level = st.sidebar.text_input("Trình độ học vấn", value="Cử nhân")
sample.append(len(education_level))  # Giả định mã hóa bằng độ dài chuỗi (có thể tùy chỉnh mã hóa khác)

# Tình trạng hôn nhân
marital_status = st.sidebar.selectbox(
    "Tình trạng hôn nhân", 
    ["Đã kết hôn", "Độc thân", "Ly hôn", "Không xác định"]
)
marital_encoded = {
    "Đã kết hôn": 0, 
    "Độc thân": 1, 
    "Ly hôn": 2, 
    "Không xác định": 3
}
sample.append(marital_encoded[marital_status])

# Thu nhập
income = st.sidebar.selectbox(
    "Thu nhập", 
    ["Dưới 40000$", "Từ 40000$ đến 60000$", "Từ 60000$ đến 80000$", "Từ 80000$ đến 120000$", "Khác"]
)
income_encoded = {
    "Dưới 40000$": 0, 
    "Từ 40000$ đến 60000$": 1, 
    "Từ 60000$ đến 80000$": 2, 
    "Từ 80000$ đến 120000$": 3, 
    "Khác": 4
}
sample.append(income_encoded[income])

# Các trường dữ liệu bổ sung
card_category = st.sidebar.selectbox(
    "Thẻ Thể loại", ["Xanh", "Bạc", "Vàng", "Bạch kim", "Khác"]
)
card_category_encoded = {
    "Xanh": 0, "Bạc": 1, "Vàng": 2, "Bạch kim": 3, "Khác": 4
}
sample.append(card_category_encoded[card_category])

months_relationship = st.sidebar.number_input(
    "Thời gian quan hệ với ngân hàng (tháng)", min_value=0, max_value=600, value=36
)
sample.append(months_relationship)

num_products = st.sidebar.number_input(
    "Tổng số sản phẩm được khách hàng nắm giữ", min_value=0, max_value=10, value=2
)
sample.append(num_products)

inactive_months = st.sidebar.number_input(
    "Số tháng không hoạt động trong 12 tháng qua", min_value=0, max_value=12, value=1
)
sample.append(inactive_months)

contacts_12_months = st.sidebar.number_input(
    "Số lượng liên lạc trong 12 tháng qua", min_value=0, max_value=50, value=3
)
sample.append(contacts_12_months)

credit_limit = st.sidebar.number_input(
    "Giới hạn tín dụng trên thẻ tín dụng ($)", min_value=0, max_value=500000, value=15000
)
sample.append(credit_limit)

revolving_balance = st.sidebar.number_input(
    "Tổng số dư luân chuyển trên thẻ tín dụng ($)", min_value=0, max_value=200000, value=5000
)
sample.append(revolving_balance)

avg_credit_open_to_buy = st.sidebar.number_input(
    "Mở để mua hạn mức tín dụng (trung bình 12 tháng qua) ($)", min_value=0, max_value=200000, value=10000
)
sample.append(avg_credit_open_to_buy)

transaction_change = st.sidebar.number_input(
    "Thay đổi về số tiền giao dịch (Quý 4 so với Quý 1)", min_value=-5.0, max_value=5.0, value=0.5, step=0.01
)
sample.append(transaction_change)

total_transaction_amount = st.sidebar.number_input(
    "Tổng số tiền giao dịch (12 tháng qua) ($)", min_value=0, max_value=1000000, value=20000
)
sample.append(total_transaction_amount)

total_transactions = st.sidebar.number_input(
    "Tổng số giao dịch (12 tháng qua)", min_value=0, max_value=1000, value=50
)
sample.append(total_transactions)

transaction_count_change = st.sidebar.number_input(
    "Thay đổi trong số lượng giao dịch (Q4 so với Q1)", min_value=-5.0, max_value=5.0, value=0.2, step=0.01
)
sample.append(transaction_count_change)

card_utilization = st.sidebar.number_input(
    "Tỷ lệ sử dụng thẻ trung bình (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1
)
sample.append(card_utilization)

if st.sidebar.button("Dự đoán"):
    sample_scaled = scaler.transform([sample])
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0, 1]
    result = "Khách hàng rời bỏ" if prediction == 1 else "Khách hàng hiện tại"
    st.write(f"Kết quả dự đoán: {result}")
    st.write(f"Xác suất: {probability*100:.2f}%")
