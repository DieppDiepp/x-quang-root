# ==============================================================================
# --- 📝 CONFIGURATIONS ---
# ==============================================================================
import streamlit as st
import pickle
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray
import os
import glob
import tensorflow as tf # Thêm TensorFlow

# Đường dẫn tới thư mục chứa các file model
# MODELS_BASE_DIR = r"D:\venv Crawling and Scraping\EnvCrawlingData\x-quang-detect\model" # SỬA ĐỔI NẾU CẦN

# Đường dẫn tương đối tính từ thư mục chứa app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_BASE_DIR = os.path.join(BASE_DIR, "model")
IMAGES_BASE_FOLDER_PATH = os.path.join(BASE_DIR, "test_photo")
NORMAL_FOLDER_NAME = "NORMAL"
PNEUMONIA_FOLDER_NAME = "PNEUMONIA"
MEME_IMAGE_PATH = os.path.join(BASE_DIR, "meme.jpg")

# Kích thước ảnh mặc định cho các model Sklearn (LBP/HOG)
SKLEARN_IMG_SIZE = 227
KERAS_IMG_SIZE = 256 # Kích thước ảnh cho model Keras

# Danh sách các model có sẵn để chọn
AVAILABLE_MODELS = {
    "Xception (Keras) F1 - 0.82": {
        "file": "trained_chest_xray_model.keras", 
        "type": "keras", 
        "img_size_needed": KERAS_IMG_SIZE
    },
    "SVM (LBP) F1 - 0.74": {
        "file": "svm_lbp_model_package.pkl", 
        "type": "sklearn_lbp", 
        "img_size_needed": SKLEARN_IMG_SIZE
    },
    "Logistic Regression (LBP) F1 - 0.71": {
        "file": "log_reg_lbp_model_package.pkl", 
        "type": "sklearn_lbp", 
        "img_size_needed": SKLEARN_IMG_SIZE
    },
    "SVM (HOG) F1 - 0.62": {
        "file": "svm_hog_model_package.pkl", 
        "type": "sklearn_hog", 
        "img_size_needed": SKLEARN_IMG_SIZE
    },
    "Logistic Regression (HOG) F1 - 0.65": {
        "file": "log_reg_hog_model_package.pkl", 
        "type": "sklearn_hog", 
        "img_size_needed": SKLEARN_IMG_SIZE
    }
}

# IMAGES_BASE_FOLDER_PATH = r"D:\venv Crawling and Scraping\EnvCrawlingData\x-quang-detect\test_photo" # SỬA ĐỔI NẾU CẦN
# NORMAL_FOLDER_NAME = "NORMAL"
# PNEUMONIA_FOLDER_NAME = "PNEUMONIA"

# # Đường dẫn đến ảnh meme
# MEME_IMAGE_PATH = r"D:\venv Crawling and Scraping\EnvCrawlingData\x-quang-detect\meme.jpg" # SỬA ĐỔI NẾU CẦN

LBP_P, LBP_R, LBP_METHOD = 8, 1, 'uniform'
HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM = 9, (8, 8), (2, 2), 'L2-Hys'
VALID_IMAGE_EXTENSIONS = ("*.jpeg", "*.jpg", "*.png", "*.bmp", "*.tiff")
KERAS_PREDICTION_THRESHOLD = 0.5

# ==============================================================================
# --- 🧠 MODEL AND FEATURE FUNCTIONS ---
# ==============================================================================
# (Các hàm extract_lbp_features, extract_hog_features, cached_load_model, 
#  predict_single_image_sklearn_logic, predict_single_image_keras_logic, 
#  get_image_paths_from_folder giữ nguyên như phiên bản trước)

def extract_lbp_features(X_images, P=LBP_P, R=LBP_R, method=LBP_METHOD):
    lbp_features_list = []
    for img_item in X_images:
        if img_item.ndim == 3 and img_item.shape[2] == 3: gray = rgb2gray(img_item)
        elif img_item.ndim == 2: gray = img_item
        else: continue
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0 and gray.min() >=0.0 : gray = (gray * 255).astype(np.uint8)
            else: gray = gray.astype(np.uint8)
        lbp = local_binary_pattern(gray, P, R, method)
        n_bins = int(lbp.max() + 1); hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        lbp_features_list.append(hist)
    if not lbp_features_list: return np.array([])
    return np.array(lbp_features_list)

def extract_hog_features(X_images, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL, cells_per_block=HOG_CELLS_PER_BLOCK, block_norm=HOG_BLOCK_NORM):
    hog_features_list = []
    for img in X_images:
        if img.ndim == 3 and img.shape[2] == 3: gray = rgb2gray(img)
        elif img.ndim == 2:
            if img.dtype != np.float32 and img.dtype != np.float64:
                if img.max() > 1: gray = img.astype(np.float32) / 255.0
                else: gray = img.astype(np.float32)
            else: gray = img
        else: continue
        feat = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm=block_norm, feature_vector=True)
        hog_features_list.append(feat)
    if not hog_features_list: return np.array([])
    return np.array(hog_features_list)

@st.cache_resource
def cached_load_model(model_filepath, model_load_type):
    if not os.path.exists(model_filepath):
        st.error(f"Lỗi: File model không tồn tại: {model_filepath}.")
        return None
    try:
        if model_load_type == 'sklearn_pkl':
            with open(model_filepath, 'rb') as f:
                model_data = pickle.load(f)
        elif model_load_type == 'keras':
            model_data = tf.keras.models.load_model(model_filepath)
        else:
            st.error(f"Loại tải model không xác định: {model_load_type}")
            return None
        return model_data
    except Exception as e:
        st.error(f"Lỗi khi tải model '{os.path.basename(model_filepath)}': {e}")
        return None

def predict_single_image_sklearn_logic(image_path, sklearn_model_package, feature_extraction_type, img_size):
    img_cv = cv2.imread(image_path)
    if img_cv is None: return None, None, None
    img_resized_bgr = cv2.resize(img_cv, (img_size, img_size))
    
    features = None
    if feature_extraction_type == "lbp": features = extract_lbp_features([img_resized_bgr])
    elif feature_extraction_type == "hog": features = extract_hog_features([img_resized_bgr])
    else: return None, None, None

    if features is None or features.size == 0: return None, None, None

    scaler = sklearn_model_package['scaler']
    features_scaled = scaler.transform(features)
    
    model = sklearn_model_package['model']
    prediction_numeric = model.predict(features_scaled)[0] 
    
    label_encoder = sklearn_model_package['label_encoder']
    predicted_class_name = label_encoder.inverse_transform([prediction_numeric])[0] 
    
    confidence_score = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        try:
            class_index_in_proba = list(label_encoder.classes_).index(predicted_class_name)
            confidence_score = probabilities[class_index_in_proba]
        except ValueError:
            if prediction_numeric < len(probabilities): confidence_score = probabilities[prediction_numeric]
    return predicted_class_name, confidence_score, prediction_numeric

def predict_single_image_keras_logic(image_path, keras_model, keras_specific_img_size):
    img_cv = cv2.imread(image_path)
    if img_cv is None: 
        st.error(f"Keras: Không thể đọc ảnh: {image_path}")
        return None, None, None
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_resized_rgb = cv2.resize(img_rgb, (keras_specific_img_size, keras_specific_img_size))
    img_normalized = img_resized_rgb / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    try:
        probability_class1 = keras_model.predict(img_batch)[0][0]
    except Exception as e:
        st.error(f"Lỗi khi dự đoán bằng model Keras: {e}")
        return None, None, None

    prediction_numeric = 1 if probability_class1 >= KERAS_PREDICTION_THRESHOLD else 0
    predicted_class_name = PNEUMONIA_FOLDER_NAME if prediction_numeric == 1 else NORMAL_FOLDER_NAME
    confidence_score = probability_class1 if prediction_numeric == 1 else (1.0 - probability_class1)
        
    return predicted_class_name, confidence_score, prediction_numeric

def get_image_paths_from_folder(folder_path, extensions=VALID_IMAGE_EXTENSIONS):
    image_files = [];
    if not os.path.isdir(folder_path): return []
    for ext in extensions: image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    return sorted([os.path.basename(p) for p in image_files])
# ==============================================================================
# --- 🖼️ STREAMLIT UI ---
# ==============================================================================

st.set_page_config(page_title="X-Ray Classification Demo", layout="wide")
st.title("️️🔬 Demo Phân Loại Ảnh X-Quang Ngực")
st.markdown("Chọn mô hình và ảnh từ thanh bên để bắt đầu.")

# --- Sidebar ---
st.sidebar.header("⚙️ Cấu Hình & Chọn Ảnh")

# 1. Chọn Model
selected_model_display_name = st.sidebar.selectbox(
    "1. Chọn mô hình để dự đoán:",
    list(AVAILABLE_MODELS.keys())
)
selected_model_info = AVAILABLE_MODELS[selected_model_display_name]
selected_model_filename = selected_model_info["file"]
model_load_type_for_cache = 'keras' if selected_model_info["type"] == 'keras' else 'sklearn_pkl'
required_img_size = selected_model_info["img_size_needed"]

model_filepath_to_load = os.path.join(MODELS_BASE_DIR, selected_model_filename)
loaded_model_object = cached_load_model(model_filepath_to_load, model_load_type_for_cache)

if loaded_model_object is None:
    st.error(f"Không thể tải model '{selected_model_display_name}'.")
    st.stop()
# else:
    # st.sidebar.success(f"Đang dùng: {selected_model_display_name}") # Có thể bỏ nếu không muốn thông báo liên tục

# 2. Chọn Ảnh
st.sidebar.markdown("---")
st.sidebar.header("🖼️ Chọn Ảnh Để Dự Đoán")

normal_folder_full_path = os.path.join(IMAGES_BASE_FOLDER_PATH, NORMAL_FOLDER_NAME)
pneumonia_folder_full_path = os.path.join(IMAGES_BASE_FOLDER_PATH, PNEUMONIA_FOLDER_NAME)
normal_image_files = get_image_paths_from_folder(normal_folder_full_path)
pneumonia_image_files = get_image_paths_from_folder(pneumonia_folder_full_path)

if not normal_image_files and not pneumonia_image_files:
    st.sidebar.error(f"Không tìm thấy ảnh mẫu trong '{IMAGES_BASE_FOLDER_PATH}'.")
    st.stop()

category_options = []
if normal_image_files: category_options.append(NORMAL_FOLDER_NAME)
if pneumonia_image_files: category_options.append(PNEUMONIA_FOLDER_NAME)

if not category_options: 
    st.sidebar.error("Không có danh mục ảnh.")
    st.stop()

selected_category = st.sidebar.selectbox("2. Chọn loại ảnh (Nhãn thật):", category_options)

current_image_files = []
current_folder_path = ""
if selected_category == NORMAL_FOLDER_NAME: 
    current_image_files, current_folder_path = normal_image_files, normal_folder_full_path
elif selected_category == PNEUMONIA_FOLDER_NAME: 
    current_image_files, current_folder_path = pneumonia_image_files, pneumonia_folder_full_path

if not current_image_files: 
    selected_image_filename = None
    st.sidebar.warning(f"Không có ảnh trong thư mục '{selected_category}'.")
else: 
    selected_image_filename = st.sidebar.selectbox(
        f"3. Chọn một ảnh từ '{selected_category}':", 
        current_image_files
    )

# THÊM ẢNH MEME VÀO SIDEBAR
st.sidebar.markdown("---") # Thêm dòng kẻ phân cách cho đẹp
if os.path.exists(MEME_IMAGE_PATH):
    st.sidebar.image(MEME_IMAGE_PATH, caption="Just for fun!", use_container_width=True)
else:
    st.sidebar.caption("Meme image not found.") # Thông báo nhẹ nhàng nếu không thấy ảnh

# --- Khu vực hiển thị chính ---
# (Code khu vực hiển thị chính giữ nguyên như phiên bản trước)
if selected_image_filename:
    selected_image_full_path = os.path.join(current_folder_path, selected_image_filename)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ảnh Được Chọn:")
        try:
            image_display_cv = cv2.imread(selected_image_full_path)
            image_display_rgb = cv2.cvtColor(image_display_cv, cv2.COLOR_BGR2RGB)
            st.image(image_display_rgb, caption=f"Nhãn thật: {selected_category} - File: {selected_image_filename}", use_container_width=True)
        except Exception as e: st.error(f"Không thể hiển thị ảnh: {e}")

    with col2:
        model_type_display = selected_model_info["type"].replace("sklearn_", "").upper()
        st.subheader(f"Kết Quả Model: {selected_model_display_name}")
        
        if st.button(f"⚡ Phân loại ảnh: {selected_image_filename}", type="primary", use_container_width=True):
            with st.spinner("⏳ Đang phân loại..."):
                predicted_class_str, confidence, predicted_class_numeric = None, None, None
                
                if selected_model_info["type"] == 'keras':
                    predicted_class_str, confidence, predicted_class_numeric = predict_single_image_keras_logic(
                        selected_image_full_path, 
                        loaded_model_object,
                        keras_specific_img_size=required_img_size
                    )
                elif selected_model_info["type"].startswith('sklearn_'):
                    feature_type = selected_model_info["type"].split('_')[-1]
                    predicted_class_str, confidence, predicted_class_numeric = predict_single_image_sklearn_logic(
                        selected_image_full_path, 
                        loaded_model_object,
                        feature_extraction_type=feature_type,
                        img_size=required_img_size
                    )
            
            if predicted_class_str is not None:
                st.markdown("#### Chẩn đoán của mô hình:")
                if predicted_class_str == PNEUMONIA_FOLDER_NAME: st.error(f"**Kết quả:** {predicted_class_str} 😞")
                else: st.success(f"**Kết quả:** {predicted_class_str} 😊")
                if confidence is not None: st.info(f"**Độ tin cậy:** {confidence:.2%}")
                else: st.info("Không có thông tin độ tin cậy.")
                st.markdown("---")
                st.markdown("#### Đánh giá độ chính xác:")
                true_label = selected_category
                predicted_label_for_eval = ""
                if predicted_class_numeric == 1: predicted_label_for_eval = PNEUMONIA_FOLDER_NAME
                elif predicted_class_numeric == 0: predicted_label_for_eval = NORMAL_FOLDER_NAME
                else: predicted_label_for_eval = predicted_class_str 
                if predicted_label_for_eval == true_label: st.balloons(); st.success(f"🎯 **Phân loại ĐÚNG!**")
                else: st.error(f"⚠️ **Phân loại SAI.**")
                st.write(f"(Ảnh gốc: '{true_label}'. Model dự đoán: '{predicted_label_for_eval}')")
            else: st.error("Không thể phân loại ảnh.")
        else: st.write("Nhấn nút 'Phân loại ảnh' để xem kết quả.")
else: st.info("Vui lòng chọn một ảnh từ thanh bên để bắt đầu.")

st.markdown("---")
st.markdown("👩‍💻 *Ứng dụng demo streamlit bởi Đắc Nguyên.*")