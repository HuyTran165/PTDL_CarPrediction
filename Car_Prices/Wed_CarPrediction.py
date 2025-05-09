import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

# Dictionary definitions for categorical features
# Brand dictionary
brand_dic = {'Audi': 0, 'BMW': 1, 'Mercedes-Benz': 2, 'Mitsubishi': 3, 'Renault': 4, 'Toyota': 5, 'Volkswagen': 6}

# Body dictionary
body_dic = {'crossover': 0, 'hatch': 1, 'other': 2, 'sedan': 3, 'vagon': 4, 'van': 5}

# Engine type dictionary
engine_type_dic = {'Diesel': 0, 'Gas': 1, 'Other': 2, 'Petrol': 3}

# Registration dictionary
registration_dic = {'no': 0, 'yes': 1}

# Model dictionary
model_dic = {'1 Series': 0, '100': 1, '11': 2, '116': 3, '118': 4, '120': 5, '19': 6, '190': 7, '200': 8, '210': 9, '220': 10,
'230': 11, '25': 12, '250': 13, '300': 14, '316': 15, '318': 16, '320': 17, '323': 18, '325': 19, '328': 20, '330': 21,
'335': 22, '4 Series Gran Coupe': 23, '428': 24, '4Runner': 25, '5 Series': 26, '5 Series GT': 27, '520': 28, '523': 29,
'524': 30, '525': 31, '528': 32, '530': 33, '535': 34, '540': 35, '545': 36, '550': 37, '6 Series Gran Coupe': 38,
'630': 39, '640': 40, '645': 41, '650': 42, '730': 43, '735': 44, '740': 45, '745': 46, '750': 47, '760': 48, '80': 49,
'9': 50, '90': 51, 'A 140': 52, 'A 150': 53, 'A 170': 54, 'A 180': 55, 'A1': 56, 'A3': 57, 'A4': 58, 'A4 Allroad': 59,
'A5': 60, 'A6': 61, 'A6 Allroad': 62, 'A7': 63, 'A8': 64, 'ASX': 65, 'Amarok': 66, 'Auris': 67, 'Avalon': 68,
'Avensis': 69, 'Aygo': 70, 'B 170': 71, 'B 180': 72, 'B 200': 73, 'Beetle': 74, 'Bora': 75, 'C-Class': 76, 'CL 180': 77,
'CL 500': 78, 'CL 55 AMG': 79, 'CL 550': 80, 'CL 63 AMG': 81, 'CLA 200': 82, 'CLA 220': 83, 'CLA-Class': 84,
'CLC 180': 85, 'CLC 200': 86, 'CLK 200': 87, 'CLK 220': 88, 'CLK 230': 89, 'CLK 240': 90, 'CLK 280': 91, 'CLK 320': 92,
'CLK 430': 93, 'CLS 350': 94, 'CLS 400': 95, 'CLS 500': 96, 'CLS 63 AMG': 97, 'Caddy': 98, 'Camry': 99, 'Captur': 100,
'Caravelle': 101, 'Carina': 102, 'Carisma': 103, 'Celica': 104, 'Clio': 105, 'Colt': 106, 'Corolla': 107,
'Corolla Verso': 108, 'Cross Touran': 109, 'Dokker': 110, 'Duster': 111, 'E-Class': 112, 'Eclipse': 113, 'Eos': 114,
'Espace': 115, 'FJ Cruiser': 116, 'Fluence': 117, 'Fortuner': 118, 'G 320': 119, 'G 350': 120, 'G 500': 121,
'G 55 AMG': 122, 'G 63 AMG': 123, 'GL 320': 124, 'GL 350': 125, 'GL 420': 126, 'GL 450': 127, 'GL 500': 128,
'GL 550': 129, 'GLC-Class': 130, 'GLE-Class': 131, 'GLK 220': 132, 'GLK 300': 133, 'GLS 350': 134, 'GLS 400': 135,
'Galant': 136, 'Golf GTI': 137, 'Golf II': 138, 'Golf III': 139, 'Golf IV': 140, 'Golf Plus': 141, 'Golf V': 142,
'Golf VI': 143, 'Golf VII': 144, 'Golf Variant': 145, 'Grand Scenic': 146, 'Grandis': 147, 'Hiace': 148,
'Highlander': 149, 'Hilux': 150, 'I3': 151, 'IQ': 152, 'Jetta': 153, 'Kangoo': 154, 'Koleos': 155, 'L 200': 156,
'LT': 157, 'Laguna': 158, 'Lancer': 159, 'Lancer Evolution': 160, 'Lancer X': 161, 'Lancer X Sportback': 162,
'Land Cruiser 100': 163, 'Land Cruiser 105': 164, 'Land Cruiser 200': 165, 'Land Cruiser 76': 166, 'Land Cruiser 80': 167,
'Land Cruiser Prado': 168, 'Latitude': 169, 'Logan': 170, 'Lupo': 171, 'M5': 172, 'M6': 173, 'MB': 174, 'ML 250': 175,
'ML 270': 176, 'ML 280': 177, 'ML 320': 178, 'ML 350': 179, 'ML 400': 180, 'ML 430': 181, 'ML 500': 182, 'ML 550': 183,
'ML 63 AMG': 184, 'Master': 185, 'Matrix': 186, 'Megane': 187, 'Modus': 188, 'Multivan': 189, 'New Beetle': 190,
'Outlander': 191, 'Outlander XL': 192, 'Pajero': 193, 'Pajero Pinin': 194, 'Pajero Sport': 195, 'Pajero Wagon': 196,
'Passat B3': 197, 'Passat B4': 198, 'Passat B5': 199, 'Passat B6': 200, 'Passat B7': 201, 'Passat B8': 202,
'Passat CC': 203, 'Phaeton': 204, 'Pointer': 205, 'Polo': 206, 'Previa': 207, 'Prius': 208, 'Q3': 209, 'Q5': 210,
'Q7': 211, 'R 320': 212, 'R8': 213, 'Rav 4': 214, 'S 140': 215, 'S 250': 216, 'S 300': 217, 'S 320': 218, 'S 350': 219,
'S 400': 220, 'S 430': 221, 'S 500': 222, 'S 550': 223, 'S 600': 224, 'S 63 AMG': 225, 'S 65 AMG': 226, 'S4': 227,
'S5': 228, 'S8': 229, 'SL 500 (550)': 230, 'SL 55 AMG': 231, 'SLK 200': 232, 'SLK 350': 233, 'Sandero': 234,
'Sandero StepWay': 235, 'Scenic': 236, 'Scion': 237, 'Scirocco': 238, 'Sequoia': 239, 'Sharan': 240, 'Sienna': 241,
'Smart': 242, 'Space Star': 243, 'Space Wagon': 244, 'Sprinter 208': 245, 'Sprinter 210': 246, 'Sprinter 211': 247,
'Sprinter 212': 248, 'Sprinter 213': 249, 'Sprinter 311': 250, 'Sprinter 312': 251, 'Sprinter 313': 252,
'Sprinter 315': 253, 'Sprinter 316': 254, 'Sprinter 318': 255, 'Sprinter 319': 256, 'Symbol': 257, 'Syncro': 258,
'T3 (Transporter)': 259, 'T4 (Transporter)': 260, 'T4 (Transporter) ': 261, 'T5 (Transporter)': 262,
'T5 (Transporter) ': 263, 'T6 (Transporter)': 264, 'T6 (Transporter) ': 265, 'TT': 266, 'Tacoma': 267, 'Tiguan': 268,
'Touareg': 269, 'Touran': 270, 'Trafic': 271, 'Tundra': 272, 'Up': 273, 'V 250': 274, 'Vaneo': 275, 'Vento': 276,
'Venza': 277, 'Viano': 278, 'Virage': 279, 'Vista': 280, 'Vito': 281, 'X1': 282, 'X3': 283, 'X5': 284, 'X5 M': 285,
'X6': 286, 'X6 M': 287, 'Yaris': 288, 'Z3': 289, 'Z4': 290}

# Lists of categorical labels
brand_list = ['Audi', 'BMW', 'Mercedes-Benz', 'Mitsubishi', 'Renault', 'Toyota', 'Volkswagen']
body_list = ['crossover', 'hatch', 'other', 'sedan', 'vagon', 'van']
engine_type_list = ['Diesel', 'Gas', 'Other', 'Petrol']
registration_list = ['yes', 'no']

# Set page configuration
st.set_page_config(
    page_title='Used Car Price Predictor',
    page_icon='🚘',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for improved UI
st.markdown("""
<style>
    /* Main styles */
    .main {
        background-color: #f8f9fa;
        padding: 20px;
    }
    
    /* Header styles */
    .header-container {
        background: linear-gradient(135deg, #051937, #004d7a);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Card styles */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .card-header {
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 10px;
        margin-bottom: 15px;
        font-weight: 600;
        color: #0a3d62;
    }
    
    /* Form styles */
    .stButton>button {
        background-color: #0a3d62;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #0c2d48;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Result box styles */
    .result-container {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    .price-value {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 15px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Footer styles */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    /* Responsive adjustments */
    @media screen and (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .price-value {
            font-size: 2.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('D:\DOWNLOAD\CAR_ANYLIST\Car_Prices\Car_cleaned_with_Model.csv')

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("D:/DOWNLOAD/CAR_ANYLIST/Car_Prices/rf1_base_rf.pkl")

# Function to find models for a specific brand
def find_model(brand, car_data):
    model = car_data[car_data['Brand'] == brand]['Model']
    return list(model)

# Main application function
def main():
    # Header section
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🚘 Used Car Price Predictor</div>
        <div class="header-subtitle">Nhận ước tính giá trị thị trường chính xác cho xe đã qua sử dụng của bạn</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Loading data and model
    with st.spinner('🔄 Đang khởi tạo hệ thống...'):
        car = load_data()
        model = load_model()
    
    # Create sidebar for app navigation
    with st.sidebar:
        st.image("D:\DOWNLOAD\CAR_ANYLIST\Data_Raw\car.jpg", width=100)
        st.title("Trang ChỦ")
        pages = ["Predict Price", "How It Works", "About"]
        choice = st.radio("Go to", pages)
    
    # Prediction page
    if choice == "Predict Price":
        # Create tabs for input organization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📋 Thông tin xe</div>', unsafe_allow_html=True)
            
            brand_inp = st.selectbox(
                "Hãng Xe",
                options=brand_list,
                help="Chọn nhà sản xuất xe"
            )
            
            # Model selection based on brand
            if brand_inp == 'Audi':
                model_options = find_model('Audi', car)
                model_inp = st.selectbox('Audi Model', options=model_options)
            elif brand_inp == 'Mitsubishi':
                model_options = find_model('Mitsubishi', car)
                model_inp = st.selectbox('Mitsubishi Model', options=model_options)
            elif brand_inp == 'Renault':   
                model_options = find_model('Renault', car)
                model_inp = st.selectbox('Renault Model', options=model_options)
            elif brand_inp == 'Toyota':
                model_options = find_model('Toyota', car)
                model_inp = st.selectbox('Toyota Model', options=model_options)
            elif brand_inp == 'BMW':
                model_options = find_model('BMW', car)
                model_inp = st.selectbox('BMW Model', options=model_options)
            elif brand_inp == 'Mercedes-Benz':
                model_options = find_model('Mercedes-Benz', car)
                model_inp = st.selectbox('Mercedes-Benz Model', options=model_options)
            elif brand_inp == 'Volkswagen':
                model_options = find_model('Volkswagen', car)
                model_inp = st.selectbox('Volkswagen Model', options=model_options)
            
            year = st.slider(
                "Năm Sản Xuất",
                min_value=1980,
                max_value=2020,
                value=2010,
                help="Năm sản xuất xe"
            )
            
            body_type = st.selectbox(
                "Kiểu Xe",
                options=body_list,
                help="Kiểu dáng thân xe"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">⚙️ Thông số kỹ thuật</div>', unsafe_allow_html=True)
            
            mileage = st.number_input(
                "Mileage (Dặm)",
                min_value=0,
                value=50000,
                step=1000,
                help="Tổng quãng đường xe đã đi được"
            )
            
            engineV = st.number_input(
                "Dung Tích Động Cơ (L)",
                min_value=0.5,
                max_value=6.4,
                value=2.0,
                step=0.1,
                help="Thể tích của động cơ ô tô tính bằng lít"
            )
            
            engine_type = st.selectbox(
                "Loại Nhiên Liệu",
                options=engine_type_list,
                help="Loại nhiên liệu xe sử dụng"
            )
            
            regis = st.selectbox(
                "Trạng thái đăng ký",
                options=registration_list,
                help="Chiếc xe đã được đăng ký chưa?"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Convert inputs to numerical values for prediction
        brand = brand_dic[brand_inp]
        body = body_dic[body_type]
        engine = engine_type_dic[engine_type]
        registration = registration_dic[regis]
        model_num = model_dic[model_inp]
        
        # Create prediction button
        st.markdown('<div style="padding: 20px 0;">', unsafe_allow_html=True)
        predict_pressed = st.button('TÍNH TOÁN GIÁ ƯỚC TÍNH', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Make prediction when button is pressed
        if predict_pressed:
            try:
                # Prepare input array
                input_array = np.array([[mileage, engineV, year, brand, body, engine, registration, model_num]])
                
                # Make prediction
                with st.spinner("🧮Đang tính giá..."):
                    prediction = model.predict(input_array)
                    price = round(float(prediction[0]), 2)
                
                if price < 0:
                    st.error("⚠️Giá tính toán là số âm. Điều này cho thấy sự kết hợp đầu vào bất thường. Vui lòng xác minh đầu vào của bạn.")
                else:
                    # Display prediction result
                    st.markdown(f"""
                    <div class="result-container">
                        <h3>Giá ước tính</h3>
                        <div class="price-value">${price:,.2f}</div>
                        <p>Dựa trên các thông số kỹ thuật bạn cung cấp</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                    
                    # Display comparable cars
                    st.markdown('<div class="card" style="margin-top: 30px;">', unsafe_allow_html=True)
                    st.markdown('<div class="card-header">🔍 Bối cảnh thị trường</div>', unsafe_allow_html=True)
                    st.write(f"Ước tính này định vị của bạn {year} {brand_inp} {model_inp} trên thị trường như sau:")
                    
                    price_range = [price * 0.9, price * 1.1]
                    st.write(f"• Những chiếc xe tương tự thường được bán với giá ${price_range[0]:,.2f} và ${price_range[1]:,.2f}")
                    st.write(f"• Số dặm của {mileage} dặm là một yếu tố quan trọng trong định giá này")
                    st.write(f"• {year} các mô hình với {engineV}L {engine_type} động cơ đã cho thấy nhu cầu ổn định")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")
                st.write("Vui lòng kiểm tra giá trị đầu vào của bạn và thử lại.")
    
    # How It Works page
    elif choice == "How It Works":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🔍 Dự đoán của chúng tôi hoạt động như thế nào</div>', unsafe_allow_html=True)
        st.write("""
        Công cụ Dự đoán giá xe của chúng tôi sử dụng thuật toán học máy tinh vi có tên là Random Forest để ước tính giá trị thị trường của xe đã qua sử dụng. Cách thức hoạt động như sau:

1. **Thu thập dữ liệu**: Chúng tôi đã phân tích hàng nghìn danh sách xe đã qua sử dụng để xây dựng mô hình dự đoán của mình.

2. **Xử lý tính năng**: Hệ thống xem xét nhiều yếu tố ảnh hưởng đến giá trị của xe:
- Thương hiệu và mẫu xe cụ thể
- Năm sản xuất
- Số km đã đi (xe đã đi được bao nhiêu dặm)
- Thể tích động cơ và loại nhiên liệu
- Kiểu thân xe
- Tình trạng đăng ký

3. **Thuật toán dự đoán**: Mô hình Random Forest của chúng tôi phân tích các yếu tố này cùng nhau để xác định giá trị thị trường có khả năng xảy ra nhất.

4. **Kết quả**: Bạn nhận được ước tính chính xác dựa trên điều kiện thị trường hiện tại và dữ liệu bán hàng lịch sử.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">💡 Mẹo để dự đoán chính xác</div>', unsafe_allow_html=True)
        st.write("""
        Để có được ước tính chính xác nhất có thể:

- Cung cấp thông tin về quãng đường đi được chính xác
- Chọn đúng mẫu xe của bạn
- Chính xác về thông số kỹ thuật của động cơ
- Cân nhắc rằng các yếu tố như tình trạng xe, lịch sử tai nạn và các tính năng bổ sung không được bao gồm trong ước tính cơ bản này
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # About page
    elif choice == "About":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">ℹ️ Giới thiệu về công cụ này</div>', unsafe_allow_html=True)
        st.write("""
        Công cụ Dự đoán giá xe này được phát triển để giúp chủ xe, người mua và người bán xe có được ước tính đáng tin cậy về giá trị xe đã qua sử dụng.

**Mục đích**:
- Giúp người bán đặt giá chào bán hợp lý
- Hỗ trợ người mua đánh giá xem giá niêm yết có hợp lý không
- Hỗ trợ đánh giá giá trị bảo hiểm
- Hỗ trợ lập kế hoạch tài chính liên quan đến tài sản xe

**Lưu ý quan trọng**: Mặc dù dự đoán của chúng tôi dựa trên phân tích dữ liệu mở rộng, giá trị thị trường thực tế của một chiếc xe cụ thể có thể thay đổi tùy thuộc vào các yếu tố bổ sung như:
- Tình trạng vật lý thực tế
- Lịch sử dịch vụ
- Nhu cầu thị trường địa phương
- Biến động theo mùa
- Các tính năng hoặc sửa đổi đặc biệt
- Màu sắc và tình trạng thẩm mỹ

Luôn tham khảo ý kiến ​​của một người thẩm định chuyên nghiệp để có định giá dứt khoát, đặc biệt là đối với các giao dịch có giá trị cao hoặc mục đích bảo hiểm.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© Công cụ dự đoán giá xe năm 2025 | Không dành cho mục đích thương mại | Giá trị chỉ là ước tính</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()