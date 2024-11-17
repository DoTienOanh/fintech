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

# Thêm tính năng dự đoán cho khách hàng mới
st.subheader("Dự đoán cho khách hàng mới")

# Thanh kéo nằm ở sidebar
st.sidebar.header("Thông tin khách hàng")
sample = []
for feature in X.columns:
    value = st.sidebar.slider(
        f"{feature}", 
        float(X[feature].min()), 
        float(X[feature].max()), 
        float(X[feature].mean())
    )
    sample.append(value)

if st.sidebar.button("Dự đoán"):
    sample_scaled = scaler.transform([sample])
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0, 1]
    result = "Khách hàng rời bỏ" if prediction == 1 else "Khách hàng hiện tại"
    st.write(f"Kết quả dự đoán: {result}")
    st.write(f"Xác suất: {probability:.4f}*100")
