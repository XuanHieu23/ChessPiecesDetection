# ChessPiecesDetection
*Luồng hoạt động: Input Image → Preprocess → YOLO Inference → Post-process → Metrics Calculation → Visualization → Output
1. CORE MODULES 
File: src/core/model_loader.py 
-Functions:
load_chess_yolo_model(weights_path, device='cuda') - Load YOLO model cho chess detection
setup_chess_classes() - Định nghĩa 13 classes chess pieces
get_chess_model_config() - Cấu hình cho chess pieces detection

File: src/core/preprocessor.py 
-Functions:
preprocess_chess_image(image, img_size=640) - Chuẩn hóa ảnh
enhance_chess_details(image) - Tăng cường chi tiết cho quân cờ nhỏ
detect_chessboard_edges(image) - Phát hiện edges của bàn cờ

File: src/core/post_processor.py 
-Functions:
post_process_chess_detections(predictions, conf_threshold=0.3) - Xử lý kết quả cho ảnh
filter_chess_pieces_by_chessboard(detections, chessboard_roi) - Lọc trong vùng bàn cờ
validate_chess_position(detections) - Validate vị trí quân cờ hợp lệ

2. DATA MODULES 
File: src/data/chess_data_loader.py 
-Functions:
load_chess_images(data_path) - Load ảnh từ dataset
load_chess_annotations(annotation_path) - Load annotations cho ảnh
split_chess_image_dataset(images, annotations) - Chia train/val/test cho ảnh
create_chess_data_loader(batch_size=16) - Tạo data loader cho training

File: src/data/chess_augmentation.py 
-Functions:
augment_chess_image(image, bboxes) - Augmentation cho ảnh cờ vua
apply_chess_specific_augmentations() - Augmentation đặc thù (xoay bàn cờ, thay đổi lighting)
generate_chess_training_pairs() - Tạo cặp ảnh training

VISUALIZATION MODULES
File: src/visualization/chess_drawer.py 
-Functions:
draw_chess_detections(image, detections) - Vẽ bounding boxes lên ẢNH
annotate_chess_pieces(image, detections) - Ghi nhãn quân cờ
draw_chessboard_coordinates(image) - Vẽ tọa độ bàn cờ (a1-h8)

File: src/visualization/chess_display.py 
-Functions:
display_chess_detection_result(image, detections) - Hiển thị kết quả detection
save_annotated_image(image, output_path, detections) - Lưu ảnh đã annotated
create_chess_detection_report_image(detections) - Tạo ảnh báo cáo kết quả

3. UTILITIES MODULES 
File: src/utils/chess_metrics.py 
-Functions:
calculate_image_detection_metrics(detections, ground_truth) - Tính metrics cho ảnh
evaluate_model_on_test_set(model, test_images) - Đánh giá trên test set ảnh
compute_confusion_matrix_chess() - Ma trận nhầm lẫn cho các quân cờ

File: src/utils/chess_evaluator.py
-Functions:
test_single_chess_image(model, image_path) - Test trên ảnh đơn
batch_test_chess_images(model, image_dir) - Test batch ảnh
generate_detection_statistics() - Thống kê detection

File: src/utils/chess_utils.py 
-Functions:
convert_detections_to_csv(detections, image_id) - Xuất kết quả sang CSV
save_detection_results(detections, output_path) - Lưu kết quả detection
load_image_with_metadata(image_path) - Load ảnh với metadata

3. TRAINING MODULES 
File: src/training/chess_trainer.py 
-Functions:
setup_chess_training_config() - Cấu hình training cho chess
train_chess_detector(model, train_loader, val_loader) - Training model
validate_chess_model(model, val_loader) - Validation trên ảnh
fine_tune_chess_model() - Fine-tune model có sẵn

File: src/training/chess_evaluation.py 
-Functions:
evaluate_trained_model(model_path, test_loader) - Đánh giá model đã train
plot_training_curves(history) - Vẽ đồ thị training
compare_chess_models(model1, model2) - So sánh các model

4. MAIN EXECUTION FOR IMAGE
File: src/chess_image_main.py 
-Functions:
process_single_chess_image(image_path) - Xử lý ảnh đơn
process_chess_image_batch(image_dir) - Xử lý batch ảnh
train_chess_detection_model() - Training model từ dataset

5. CONFIG
File: configs/chess_config.yaml- chứa tất cả cấu hình cho dự án, bao gồm đường dẫn, tham số model, tham số training, v.v.

-------------GIẢI THÍCH THUẬT NGỮ----------------
1. CORE MODULES THUẬT NGỮ:
model_loader.py:
YOLO (You Only Look Once): Model AI phát hiện vật thể trong 1 lần duyệt ảnh
Weights: File chứa thông số đã train của model
Classes: 13 loại (12 quân cờ + bàn cờ) - trắng/đen cho mỗi quân
Device ('cuda'): Sử dụng GPU để tăng tốc tính toán

preprocessor.py:
Preprocessing: Chuẩn hóa ảnh đầu vào (resize, normalize values)
Image Enhancement: Tăng cường chi tiết ảnh để model nhìn rõ hơn
Edge Detection: Phát hiện đường viền - dùng để tìm bàn cờ
Chessboard ROI: Vùng quan tâm (Region of Interest) - chỉ xử lý trong bàn cờ

post_processor.py:
Post-processing: Xử lý kết quả thô từ model
Confidence Threshold: Ngưỡng tin cậy (0.3 = 30%) - chỉ giữ detection có độ tin cậy > 30%
NMS (Non-Maximum Suppression): Kỹ thuật loại bỏ bounding box trùng nhau
Validation: Kiểm tra tính hợp lệ của vị trí quân cờ

2. DATA MODULES THUẬT NGỮ:
chess_data_loader.py:
Dataset: Bộ dữ liệu ảnh + annotations
Annotations: File ghi nhãn (vị trí, class của quân cờ)
Train/Val/Test Split: Chia dữ liệu thành 3 phần (huấn luyện/kiểm tra/đánh giá)
Data Loader: Công cụ nạp dữ liệu theo batch cho training

chess_augmentation.py:
Data Augmentation: Kỹ thuật tăng dữ liệu bằng cách biến đổi ảnh
Chess-specific Augmentations: Biến đổi đặc thù cho cờ vua (xoay bàn cờ, thay đổi ánh sáng)
Training Pairs: Cặp ảnh trước/sau augmentation

3. VISUALIZATION THUẬT NGỮ:
chess_drawer.py:
Bounding Boxes: Khung hình chữ nhật bao quanh vật thể phát hiện
Annotation: Ghi nhãn tên quân cờ + độ tin cậy
Chessboard Coordinates: Tọa độ bàn cờ (a1, b1, ..., h8)

chess_display.py:
Detection Result: Kết quả phát hiện vật thể
Annotated Image: Ảnh đã được vẽ bounding box + nhãn
Report Image: Ảnh báo cáo tổng hợp kết quả

4. METRICS & UTILITIES THUẬT NGỮ: 
chess_metrics.py:
Precision: Độ chính xác = Số detection đúng / Tổng số detection
Recall: Độ bao phủ = Số detection đúng / Tổng số vật thể thực tế
F1-Score: Trung bình điều hòa của Precision và Recall
mAP (mean Average Precision): Chỉ số quan trọng nhất trong object detection
Confusion Matrix: Ma trận hiển thị số lần nhận diện đúng/sai giữa các classes

chess_evaluator.py:
Single Image Test: Kiểm tra model trên 1 ảnh duy nhất
Batch Test: Kiểm tra model trên nhiều ảnh cùng lúc
Detection Statistics: Thống kê số lượng phát hiện theo class

chess_utils.py:
CSV Export: Xuất kết quả sang file Excel/CSV
JSON Format: Định dạng dữ liệu phổ biến để lưu kết quả
Metadata: Thông tin bổ sung về ảnh (kích thước, định dạng, etc.)

5. TRAINING MODULES THUẬT NGỮ:
chess_trainer.py:
Training Config: Cấu hình quá trình huấn luyện
Fine-tuning: Huấn luyện tiếp model đã được train sẵn
Validation: Kiểm tra model trong quá trình training
Epochs: Số lần duyệt toàn bộ dataset

chess_evaluation.py:
Training Curves: Đồ thị thể hiện tiến trình training
Model Comparison: So sánh hiệu suất giữa các model
Trained Model: Model đã hoàn thành huấn luyện

6. EXECUTION THUẬT NGỮ:
chess_image_main.py:
Single Image Processing: Xử lý 1 ảnh duy nhất
Batch Processing: Xử lý nhiều ảnh cùng lúc
Pipeline: Chuỗi các bước xử lý từ đầu đến cuối
