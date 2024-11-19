import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Đọc và xử lý dữ liệu
@st.cache_data
def load_data(file_path):
    # Đọc file CSV
    data = pd.read_csv(file_path)
    columns_to_drop = [
        'CLIENTNUM', 
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    # Loại bỏ các cột không cần thiết
    data_cleaned = data.drop(columns=columns_to_drop)
    # Mã hóa biến 'Attrition_Flag'
    data_cleaned['Attrition_Flag'] = LabelEncoder().fit_transform(data_cleaned['Attrition_Flag'])
    # Biến đổi các cột dạng phân loại sang dạng one-hot encoding
    data_encoded = pd.get_dummies(data_cleaned, drop_first=True)
    return data_encoded

# Đường dẫn mặc định
file_path = "BankChurners.csv"

# Giao diện chính
st.title("Dự đoán khả năng rời bỏ khách hàng")

# Tải dữ liệu
data = load_data(file_path)
X = data.drop(columns=['Attrition_Flag'])
y = data['Attrition_Flag']

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Áp dụng SMOTE để xử lý mất cân bằng dữ liệu
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình XGBoost
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train_scaled, y_train_resampled)

# Sidebar nhập dữ liệu
st.sidebar.header("Thông tin khách hàng mới")
sample = {}

# Thông tin nhập liệu
sample['Customer_Age'] = st.sidebar.number_input("Tuổi khách hàng", min_value=18, max_value=100, value=30)
sample['Gender_M'] = st.sidebar.selectbox("Giới tính", ["Nam", "Nữ"]) == "Nam"
income_category = st.sidebar.selectbox("Thu nhập", [
    "Dưới 40000$", "Từ 40000$ -> 60000$", "Từ 60000$ -> 80000$", "Từ 80000$ -> 120000$", "Khác"
])
sample['Income_Category'] = income_category
sample['Dependent_count'] = st.sidebar.number_input("Số người phụ thuộc", min_value=0, max_value=10, value=2)
sample['Months_on_book'] = st.sidebar.number_input("Thời gian giao dịch (tháng)", min_value=0, max_value=600, value=36)
sample['Total_Relationship_Count'] = st.sidebar.number_input("Số lượng mối quan hệ với ngân hàng", min_value=0, max_value=10, value=4)
sample['Months_Inactive_12_mon'] = st.sidebar.number_input("Số tháng không hoạt động", min_value=0, max_value=12, value=2)
sample['Contacts_Count_12_mon'] = st.sidebar.number_input("Số lần liên lạc trong 12 tháng qua", min_value=0, max_value=50, value=3)
sample['Credit_Limit'] = st.sidebar.number_input("Hạn mức tín dụng ($)", min_value=0, max_value=500000, value=15000)
sample['Total_Revolving_Bal'] = st.sidebar.number_input("Dư nợ luân chuyển ($)", min_value=0, max_value=200000, value=5000)
sample['Avg_Open_To_Buy'] = st.sidebar.number_input("Hạn mức tín dụng mở để mua ($)", min_value=0, max_value=200000, value=10000)
sample['Total_Trans_Amt'] = st.sidebar.number_input("Tổng giao dịch ($)", min_value=0, max_value=1000000, value=20000)
sample['Total_Trans_Ct'] = st.sidebar.number_input("Tổng số giao dịch", min_value=0, max_value=1000, value=50)
sample['Avg_Utilization_Ratio'] = st.sidebar.number_input("Tỷ lệ sử dụng thẻ (%)", min_value=0.0, max_value=1.0, value=0.3)
sample['Total_Ct_Chng_Q4_Q1'] = st.sidebar.number_input("Thay đổi số tiền giao dịch (Quý 4 so với Quý 1)", min_value=-1000000, max_value=1000000, value=0)

# Dự đoán
if st.sidebar.button("Dự đoán"):
    try:
        # Chuyển đổi thành DataFrame
        sample_df = pd.DataFrame([sample])

        # Xử lý các cột còn thiếu bằng cách đồng bộ với X
        sample_encoded = pd.get_dummies(sample_df)
        sample_encoded = sample_encoded.reindex(columns=X.columns, fill_value=0)

        # Chuẩn hóa dữ liệu đầu vào
        sample_scaled = scaler.transform(sample_encoded)

        # Dự đoán
        prediction = model.predict(sample_scaled)[0]
        probability = model.predict_proba(sample_scaled)[0, 1]

        result = "Khách hàng rời bỏ" if prediction == 1 else "Khách hàng hiện tại"
        st.write(f"**Kết quả dự đoán:** {result}")
        st.write(f"**Xác suất rời bỏ:** {probability * 100:.2f}%")
    except Exception as e:
        st.error(f"Lỗi trong quá trình dự đoán: {e}")
