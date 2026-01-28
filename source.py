import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import pickle
import os
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tqdm import tqdm
from datetime import datetime
# Deep Learning imports
from sklearn.svm import SVC  # ← BỔ SUNG SVM
from sklearn.svm import LinearSVC
import time
import psutil
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
device = 'mps' # 'mps' if torch.backends.mps.is_available() else 'cpu'
epochs = 100  # Số epoch mặc định cho training

class FlowExtractor:
    """Trích xuất optical flow từ video"""
    
    def __init__(self, model_path='yolov8s.pt', flow_sequence_length=15):
        self.model = YOLO(model_path)
        self.flow_sequence_length = flow_sequence_length
        self.person_flows = defaultdict(lambda: deque(maxlen=flow_sequence_length))
        self.prev_gray = None
        self.extracted_sequences = []
        
    def calculate_optical_flow(self, prev_gray, curr_gray, bbox):
        """Tính optical flow trong bounding box"""
        x1, y1, x2, y2 = map(int, bbox) # x, y, w, h = bboxes[frame_idx]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(curr_gray.shape[1], x2)
        y2 = min(curr_gray.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        prev_roi = prev_gray[y1:y2, x1:x2]
        curr_roi = curr_gray[y1:y2, x1:x2]
        
        if prev_roi.size == 0 or curr_roi.size == 0:
            return None
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_roi, curr_roi, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        # 3. Trích xuất flow bên trong Bounding Box
        # roi_flow = flow[y1:y1+y2, x1:x1+x2]
        return flow
    
    def extract_flow_features(self, flow):
        """Trích xuất đặc trưng từ optical flow"""
        if flow is None:
            return None
            
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Tùy chọn: Chuẩn hóa magnitude để làm đặc trưng đầu vào
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Đặc trưng thống kê
        features = {
            'mean_magnitude': np.mean(mag),
            'std_magnitude': np.std(mag),
            'max_magnitude': np.max(mag),
            'min_magnitude': np.min(mag),
            'mean_angle': np.mean(ang),
            'std_angle': np.std(ang),
            
            # Histogram magnitude
            'mag_hist': np.histogram(mag, bins=8, range=(0, 10))[0],
            
            # Histogram angle (8 directions)
            'ang_hist': np.histogram(ang, bins=8, range=(0, 2*np.pi))[0],
            
            # Percentiles
            'mag_percentile_25': np.percentile(mag, 25),
            'mag_percentile_50': np.percentile(mag, 50),
            'mag_percentile_75': np.percentile(mag, 75),
        }
        
        return features
    
    def process_video(self, video_path, show_video=False, label=None):
        """Xử lý một video và trích xuất sequences"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            return []
        
        self.person_flows.clear()
        self.prev_gray = None
        sequences = []
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # YOLO tracking
            results = self.model.track(
                frame, persist=True, classes=[0], verbose=False, tracker="custom_botsort.yaml", device=device
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes, track_ids, confs):
                    if show_video:
                        # --- PHẦN BỔ SUNG: VẼ BOUNDING BOX VÀ CONFIDENCE ---
                        x1, y1, x2, y2 = map(int, box)
                        label_str = f"ID:{track_id} {conf:.2f}"
                        
                        # Vẽ hình chữ nhật (BBox) - Màu xanh lá (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Vẽ nhãn (ID và Confidence)
                        cv2.putText(frame, label_str, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # --------------------------------------------------
                    
                    if self.prev_gray is not None:
                        flow = self.calculate_optical_flow(
                            self.prev_gray, curr_gray, box
                        )
                        
                        if flow is not None:
                            features = self.extract_flow_features(flow)
                            
                            if features is not None:
                                self.person_flows[track_id].append(features)
                                
                                # Nếu đủ 15 frames
                                if len(self.person_flows[track_id]) == self.flow_sequence_length:
                                    sequence = list(self.person_flows[track_id])
                                    sequences.append({
                                        'features': sequence,
                                        'label': label,
                                        'video_path': video_path,
                                        'person_id': track_id
                                    })
            if show_video:
                # Hiển thị video trực tiếp
                cv2.imshow("Violence Detection - Processing", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.prev_gray = curr_gray.copy()
        
        cap.release()
        return sequences

class DatasetBuilder:
    """Xây dựng dataset từ thư mục Violent/Non-Violent"""
    
    def __init__(self, data_dir, flow_sequence_length=15):
        """
        Args:
            data_dir: Thư mục chứa 2 folder Violent/ và Non-Violent/
            flow_sequence_length: Số frame optical flow
        """
        self.data_dir = Path(data_dir)
        self.flow_sequence_length = flow_sequence_length
        self.extractor = FlowExtractor(flow_sequence_length=flow_sequence_length)
        
    def build_dataset2(self, output_path='dataset.pkl'):
        """Xây dựng dataset từ tất cả videos"""
        
        all_sequences = []
        
        # Xử lý Violent videos
        violent_dir = self.data_dir / 'Violent'
        if violent_dir.exists():
            violent_videos = list(violent_dir.glob('*.mp4')) + \
                           list(violent_dir.glob('*.avi')) + \
                           list(violent_dir.glob('*.mov'))
            
            print(f"\n===== Xử lý {len(violent_videos)} Violent videos =====")
            for video_path in tqdm(violent_videos, desc="Violent"):
                sequences = self.extractor.process_video(str(video_path), label=1)
                all_sequences.extend(sequences)
                print(f"  {video_path.name}: {len(sequences)} sequences")
        
        # Xử lý Non-Violent videos
        non_violent_dir = self.data_dir / 'Non-Violent'
        if non_violent_dir.exists():
            non_violent_videos = list(non_violent_dir.glob('*.mp4')) + \
                               list(non_violent_dir.glob('*.avi')) + \
                               list(non_violent_dir.glob('*.mov'))
            
            print(f"\n===== Xử lý {len(non_violent_videos)} Non-Violent videos =====")
            for video_path in tqdm(non_violent_videos, desc="Non-Violent"):
                sequences = self.extractor.process_video(str(video_path), label=0)
                all_sequences.extend(sequences)
                print(f"  {video_path.name}: {len(sequences)} sequences")
        
        # Lưu dataset
        with open(output_path, 'wb') as f:
            pickle.dump(all_sequences, f)
        
        print(f"\n===== TỔNG KẾT =====")
        print(f"Tổng số sequences: {len(all_sequences)}")
        
        violent_count = sum(1 for s in all_sequences if s['label'] == 1)
        non_violent_count = sum(1 for s in all_sequences if s['label'] == 0)
        
        print(f"Violent: {violent_count}")
        print(f"Non-Violent: {non_violent_count}")
        print(f"Đã lưu dataset vào: {output_path}")
        
        return all_sequences

    def build_dataset(self, output_path='dataset.pkl', log_path='processing_log.txt'):
        """Xây dựng dataset và lưu nhật ký xử lý ra file text"""
        
        all_sequences = []
        log_entries = []
        
        # Ghi chú thời gian bắt đầu vào log
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entries.append(f"--- BẮT ĐẦU XỬ LÝ DATASET: {start_time} ---")

        def process_category(category_name, label):
            category_dir = self.data_dir / category_name
            if not category_dir.exists():
                return []
            
            videos = []
            # for ext in ['*.mp4', '*.avi', '*.mov']:
            #     videos.extend(list(category_dir.glob(ext)))
            for ext in ['*.mp4', '*.avi', '*.mov']:
                # rglob sẽ tìm xuyên qua mọi cấp thư mục con
                videos.extend(list(category_dir.rglob(ext)))
                
            print(f"\n===== Xử lý {len(videos)} {category_name} videos =====")
            log_entries.append(f"\nDanh mục: {category_name} ({len(videos)} videos)")
            
            category_sequences = []
            for video_path in tqdm(videos, desc=category_name):
                sequences = self.extractor.process_video(str(video_path), label=label)
                category_sequences.extend(sequences)
                
                # Lưu thông tin video vào log
                msg = f"  {video_path.name}: {len(sequences)} sequences trích xuất thành công"
                print(msg)
                log_entries.append(msg)
                
            return category_sequences

        # Xử lý 2 nhóm
        all_sequences.extend(process_category('violent', label=1))
        all_sequences.extend(process_category('non-violent', label=0))

        # 1. Lưu file Binary (Pickle) để máy học
        with open(output_path, 'wb') as f:
            pickle.dump(all_sequences, f)
        
        # 2. Tổng kết và Lưu file Text (Log) cho người đọc
        violent_count = sum(1 for s in all_sequences if s['label'] == 1)
        non_violent_count = sum(1 for s in all_sequences if s['label'] == 0)
        
        summary = [
            "\n" + "="*30,
            f"TỔNG KẾT QUÁ TRÌNH",
            f"Tổng số sequences: {len(all_sequences)}",
            f"Violent: {violent_count}",
            f"Non-Violent: {non_violent_count}",
            f"Thời gian kết thúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"File dataset: {output_path}",
            "="*30
        ]
        log_entries.extend(summary)

        # Ghi tất cả log vào file text
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(log_entries))

        for line in summary: print(line)
        print(f"Nhật ký xử lý đã được lưu tại: {log_path}")
        
        return all_sequences

class RandomForestClassifier:
    """Classifier học có giám sát để nhận dạng hành động"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def sequence_to_feature_vector(self, sequence):
        """
        Chuyển sequence 15 frames thành 1 feature vector
        
        Args:
            sequence: List of 15 flow features dict
            
        Returns:
            feature_vector: 1D numpy array
        """
        features = []
        
        for frame_features in sequence:
            # Đặc trưng scalar
            features.extend([
                frame_features['mean_magnitude'],
                frame_features['std_magnitude'],
                frame_features['max_magnitude'],
                frame_features['min_magnitude'],
                frame_features['mean_angle'],
                frame_features['std_angle'],
                frame_features['mag_percentile_25'],
                frame_features['mag_percentile_50'],
                frame_features['mag_percentile_75'],
            ])
            
            # Histogram
            features.extend(frame_features['mag_hist'].tolist())
            features.extend(frame_features['ang_hist'].tolist())
        
        return np.array(features)
    
    def prepare_data(self, sequences):
        """Chuẩn bị X, y từ sequences"""
        X = []
        y = []
        
        for seq in sequences:
            feature_vector = self.sequence_to_feature_vector(seq['features'])
            X.append(feature_vector)
            y.append(seq['label'])
        
        return np.array(X), np.array(y)
    
    def train(self, sequences, test_size=0.2):
        """
        Train model từ sequences có nhãn
        
        Args:
            sequences: List of sequences với label
            test_size: Tỷ lệ test set
        """
        print("\n===== CHUẨN BỊ DỮ LIỆU =====")
        X, y = self.prepare_data(sequences)
        
        print(f"Số lượng samples: {len(X)}")
        print(f"Số chiều features: {X.shape[1]}")
        print(f"Phân bố class: Violent={sum(y)}, Non-Violent={len(y)-sum(y)}")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Chuẩn hóa dữ liệu
        print("\n===== CHUẨN HÓA DỮ LIỆU =====")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Training
        print("\n===== TRAINING MODEL =====")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluation
        print("\n===== ĐÁNH GIÁ MODEL =====")
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        print("\n--- Classification Report ---")
        print(classification_report(
            y_test, y_pred,
            target_names=['Non-Violent', 'Violent']
        ))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                Predicted")
        print(f"              Non-V  Violent")
        print(f"Actual Non-V  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Violent{cm[1][0]:5d}  {cm[1][1]:5d}")
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, sequence):
        """
        Dự đoán hành động từ sequence
        
        Args:
            sequence: List of 15 flow features
            
        Returns:
            prediction: 0 (Non-Violent) hoặc 1 (Violent)
            probability: Xác suất của từng class
        """
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        
        X = self.sequence_to_feature_vector(sequence).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        return prediction, probabilities
    
    def save_model(self, model_path='action_classifier.pkl'):
        """Lưu model đã train"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }, model_path)
        print(f"Đã lưu model vào: {model_path}")
    
    def load_model(self, model_path='action_classifier.pkl'):
        """Load model đã train"""
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']
        print(f"Đã load model từ: {model_path}")

class FlowSequenceDataset(Dataset):
    """PyTorch Dataset cho optical flow sequences"""
    
    def __init__(self, sequences, reshape_for_3d=False):
        """
        Args:
            sequences: List of sequences từ DatasetBuilder
            reshape_for_3d: True nếu dùng cho 3D CNN
        """
        self.data = []
        self.labels = []
        self.reshape_for_3d = reshape_for_3d
        
        for seq in sequences:
            # Chuyển sequence thành tensor
            flow_features = self._prepare_sequence(seq['features'])
            self.data.append(flow_features)
            self.labels.append(seq['label'])
        
        self.data = torch.FloatTensor(np.array(self.data))
        self.labels = torch.LongTensor(self.labels)
        
        # Reshape cho 3D CNN nếu cần
        if reshape_for_3d:
            # (batch, 15, 25) -> (batch, 1, 15, 5, 5)
            # Reshape 25 features thành 5x5 spatial grid
            batch_size = self.data.shape[0]
            self.data = self.data.view(batch_size, 15, 5, 5)
            self.data = self.data.unsqueeze(1)  # Thêm channel dimension
    
    def _prepare_sequence(self, sequence):
        """
        Chuyển sequence features thành tensor
        
        Args:
            sequence: List of 15 flow features dict
            
        Returns:
            tensor: Shape (15, num_features)
        """
        features_per_frame = []
        
        for frame_features in sequence:
            frame_vec = [
                frame_features['mean_magnitude'],
                frame_features['std_magnitude'],
                frame_features['max_magnitude'],
                frame_features['min_magnitude'],
                frame_features['mean_angle'],
                frame_features['std_angle'],
                frame_features['mag_percentile_25'],
                frame_features['mag_percentile_50'],
                frame_features['mag_percentile_75'],
            ]
            # Thêm histograms
            frame_vec.extend(frame_features['mag_hist'].tolist())
            frame_vec.extend(frame_features['ang_hist'].tolist())
            
            features_per_frame.append(frame_vec)
        
        return np.array(features_per_frame)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CNNLSTM(nn.Module):
    """
    CNN-LSTM model cho nhận dạng hành động từ optical flow
    
    Architecture:
    - CNN: Trích xuất đặc trưng không gian từ mỗi frame
    - LSTM: Học temporal patterns qua 15 frames
    - FC: Classification layer
    """
    
    def __init__(self, input_size=25, hidden_size=128, num_layers=2, 
                 num_classes=2, dropout=0.5):
        """
        Args:
            input_size: Số features per frame (25 = 9 scalar + 8 mag_hist + 8 ang_hist)
            hidden_size: LSTM hidden size
            num_layers: Số LSTM layers
            num_classes: 2 (Violent/Non-Violent)
            dropout: Dropout rate
        """
        super(CNNLSTM, self).__init__()
        
        # CNN layers để trích xuất features từ mỗi frame
        self.cnn = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        
        # LSTM để học temporal patterns
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length=15, input_size=25)
            
        Returns:
            output: Predictions (batch_size, num_classes)
        """
        batch_size, seq_len, features = x.size()
        
        # Reshape để áp dụng CNN cho từng frame
        x = x.view(batch_size * seq_len, features)
        
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch_size * seq_len, 128)
        
        # Reshape lại cho LSTM
        cnn_out = cnn_out.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)  # (batch_size, seq_len, hidden_size*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size*2)
        
        # Classification
        output = self.fc(attended)  # (batch_size, num_classes)
        
        return output

class CNNLSTMClassifier:
    """Wrapper cho CNN-LSTM model với training và prediction"""
    
    def __init__(self, input_size=25, hidden_size=128, num_layers=2, 
                 dropout=0.5, device=None):
        """
        Args:
            input_size: Số features per frame
            hidden_size: LSTM hidden size
            num_layers: Số LSTM layers
            dropout: Dropout rate
            device: 'cuda' hoặc 'cpu'
        """
        if device is None:
            self.device = 'cpu'# device('mps' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Sử dụng device: {self.device}")
        
        self.model = CNNLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout
        ).to(self.device)
        
        self.is_trained = False
        
    def train(self, sequences, test_size=0.2, batch_size=32, 
              epochs=100, lr=0.001, weight_decay=1e-4):
        """
        Train CNN-LSTM model
        
        Args:
            sequences: List of sequences từ DatasetBuilder
            test_size: Tỷ lệ test set
            batch_size: Batch size
            epochs: Số epochs
            lr: Learning rate
            weight_decay: L2 regularization
        """
        print("\n===== CHUẨN BỊ DỮ LIỆU CNN-LSTM =====")
        
        # Tạo dataset
        labels = [seq['label'] for seq in sequences]
        train_seqs, test_seqs = train_test_split(
            sequences, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = FlowSequenceDataset(train_seqs)
        test_dataset = FlowSequenceDataset(test_seqs)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Input shape: (sequence_length=15, features={train_dataset.data.shape[2]})")
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Loss và optimizer
        # Class weights để handle imbalanced data
        train_labels = train_dataset.labels.numpy()
        class_counts = np.bincount(train_labels)
        class_weights = torch.FloatTensor(
            [len(train_labels) / (2 * c) for c in class_counts]
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5 #, verbose=True
        )
        
        print(f"\nClass weights: Non-Violent={class_weights[0]:.3f}, "
              f"Violent={class_weights[1]:.3f}")
        
        # Training loop
        print("\n===== TRAINING CNN-LSTM =====")
        best_test_acc = 0.0
        train_losses = []
        test_accs = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Evaluation
            self.model.eval()
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            test_acc = 100 * test_correct / test_total
            
            # Learning rate scheduling
            scheduler.step(test_acc)
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), 'best_cnn_lstm_model.pth')
            
            train_losses.append(avg_train_loss)
            test_accs.append(test_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {avg_train_loss:.4f} "
                      f"Train Acc: {train_acc:.2f}% "
                      f"Test Acc: {test_acc:.2f}%")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_cnn_lstm_model.pth'))
        
        # Final evaluation
        print("\n===== ĐÁNH GIÁ CUỐI CÙNG =====")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")
        
        print("\n--- Classification Report ---")
        print(classification_report(
            all_labels, all_preds,
            target_names=['Non-Violent', 'Violent']
        ))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(all_labels, all_preds)
        print(f"                Predicted")
        print(f"              Non-V  Violent")
        print(f"Actual Non-V  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Violent{cm[1][0]:5d}  {cm[1][1]:5d}")
        
        self.is_trained = True
        
        return {
            'best_test_accuracy': best_test_acc,
            'train_losses': train_losses,
            'test_accuracies': test_accs
        }
    
    def predict(self, sequence):
        """
        Dự đoán từ sequence
        
        Args:
            sequence: List of 15 flow features dict
            
        Returns:
            prediction: 0 (Non-Violent) hoặc 1 (Violent)
            probabilities: Xác suất của từng class
        """
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        
        self.model.eval()
        
        # Chuẩn bị input
        dataset = FlowSequenceDataset([{'features': sequence, 'label': 0}])
        x = dataset.data[0].unsqueeze(0).to(self.device)  # (1, 15, 25)
        
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
        
        return prediction, probabilities
    
    def save_model(self, model_path='cnn_lstm_classifier.pth'):
        """Lưu model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'is_trained': self.is_trained
        }, model_path)
        print(f"Đã lưu CNN-LSTM model vào: {model_path}")
    
    def load_model(self, model_path='cnn_lstm_classifier.pth'):
        """Load model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        self.model.eval()
        print(f"Đã load CNN-LSTM model từ: {model_path}")

class CNN3D(nn.Module):
    """
    3D CNN model cho nhận dạng hành động từ optical flow
    
    Xử lý toàn bộ sequence như một khối 3D (temporal + spatial)
    """
    
    def __init__(self, num_classes=2, dropout=0.5):
        """
        Args:
            num_classes: 2 (Violent/Non-Violent)
            dropout: Dropout rate
        """
        super(CNN3D, self).__init__()
        
        # Input shape: (batch, 1, 15, 5, 5)
        # 1 channel, 15 frames, 5x5 spatial grid
        
        # 3D Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=(1, 2, 2))  # Giữ temporal dimension
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(2, 2, 2))  # Reduce tất cả dimensions
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # Global average pooling
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 1, 15, 5, 5)
            
        Returns:
            output: Predictions (batch_size, num_classes)
        """
        x = self.conv1(x)  # (batch, 32, 15, 2, 2)
        x = self.conv2(x)  # (batch, 64, 7, 1, 1)
        x = self.conv3(x)  # (batch, 128, 1, 1, 1)
        x = self.fc(x)     # (batch, num_classes)
        
        return x

class CNN3DClassifier:
    """Wrapper cho 3D CNN model"""
    
    def __init__(self, dropout=0.5, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Sử dụng device: {self.device}")
        
        self.model = CNN3D(num_classes=2, dropout=dropout).to(self.device)
        self.is_trained = False
    
    def train(self, sequences, test_size=0.2, batch_size=32,
              epochs=100, lr=0.001, weight_decay=1e-4):
        """Train 3D CNN model"""
        print("\n===== CHUẨN BỊ DỮ LIỆU 3D CNN =====")
        
        labels = [seq['label'] for seq in sequences]
        train_seqs, test_seqs = train_test_split(
            sequences, test_size=test_size, random_state=42, stratify=labels
        )
        
        # reshape_for_3d=True để có shape (batch, 1, 15, 5, 5)
        train_dataset = FlowSequenceDataset(train_seqs, reshape_for_3d=True)
        test_dataset = FlowSequenceDataset(test_seqs, reshape_for_3d=True)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Input shape: {train_dataset.data.shape}")
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Loss và optimizer
        train_labels = train_dataset.labels.numpy()
        class_counts = np.bincount(train_labels)
        class_weights = torch.FloatTensor(
            [len(train_labels) / (2 * c) for c in class_counts]
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        print(f"\nClass weights: Non-Violent={class_weights[0]:.3f}, "
              f"Violent={class_weights[1]:.3f}")
        
        # Training loop
        print("\n===== TRAINING 3D CNN =====")
        best_test_acc = 0.0
        train_losses = []
        test_accs = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Evaluation
            self.model.eval()
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            test_acc = 100 * test_correct / test_total
            scheduler.step(test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), 'best_3dcnn_model.pth')
            
            train_losses.append(avg_train_loss)
            test_accs.append(test_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {avg_train_loss:.4f} "
                      f"Train Acc: {train_acc:.2f}% "
                      f"Test Acc: {test_acc:.2f}%")
        
        self.model.load_state_dict(torch.load('best_3dcnn_model.pth'))
        
        print("\n===== ĐÁNH GIÁ CUỐI CÙNG =====")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")
        
        print("\n--- Classification Report ---")
        print(classification_report(
            all_labels, all_preds,
            target_names=['Non-Violent', 'Violent']
        ))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(all_labels, all_preds)
        print(f"                Predicted")
        print(f"              Non-V  Violent")
        print(f"Actual Non-V  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Violent{cm[1][0]:5d}  {cm[1][1]:5d}")
        
        self.is_trained = True
        
        return {
            'best_test_accuracy': best_test_acc,
            'train_losses': train_losses,
            'test_accuracies': test_accs
        }
    
    def predict(self, sequence):
        """Dự đoán từ sequence"""
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        
        self.model.eval()
        
        dataset = FlowSequenceDataset(
            [{'features': sequence, 'label': 0}], 
            reshape_for_3d=True
        )
        x = dataset.data[0].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
        
        return prediction, probabilities
    
    def save_model(self, model_path='3dcnn_classifier.pth'):
        """Lưu model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'is_trained': self.is_trained
        }, model_path)
        print(f"Đã lưu 3D CNN model vào: {model_path}")
    
    def load_model(self, model_path='3dcnn_classifier.pth'):
        """Load model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        self.model.eval()
        print(f"Đã load 3D CNN model từ: {model_path}")

class EnsembleClassifier:
    """
    Ensemble kết hợp nhiều models
    
    Strategies:
    - Voting: Majority voting
    - Weighted: Weighted average based on confidence
    - Stacking: Meta-learner
    """
    
    def __init__(self, models, weights=None, strategy='weighted'):
        """
        Args:
            models: Dict of models {'name': model_instance}
            weights: Dict of weights {'name': weight} (for weighted strategy)
            strategy: 'voting', 'weighted', hoặc 'stacking'
        """
        self.models = models
        self.strategy = strategy
        
        if weights is None:
            # Equal weights
            self.weights = {name: 1.0/len(models) for name in models.keys()}
        else:
            # Normalize weights
            total = sum(weights.values())
            self.weights = {k: v/total for k, v in weights.items()}
        
        print(f"Ensemble Strategy: {strategy}")
        print(f"Models: {list(models.keys())}")
        print(f"Weights: {self.weights}")
    
    def predict(self, sequence):
        """
        Dự đoán bằng ensemble
        
        Args:
            sequence: List of 15 flow features
            
        Returns:
            prediction: 0 (Non-Violent) hoặc 1 (Violent)
            probabilities: Ensemble probabilities
        """
        predictions = []
        all_probs = []
        
        # Lấy predictions từ tất cả models
        for name, model in self.models.items():
            pred, probs = model.predict(sequence)
            predictions.append(pred)
            all_probs.append(probs)
        
        if self.strategy == 'voting':
            # Majority voting
            final_pred = int(mode(predictions, keepdims=True)[0][0])
            # Average probabilities
            final_probs = np.mean(all_probs, axis=0)
            
        elif self.strategy == 'weighted':
            # Weighted average of probabilities
            weighted_probs = np.zeros(2)
            for (name, model), probs in zip(self.models.items(), all_probs):
                weighted_probs += self.weights[name] * probs
            
            final_probs = weighted_probs
            final_pred = np.argmax(final_probs)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return final_pred, final_probs
    
    def evaluate(self, sequences):
        """Đánh giá ensemble trên test set"""
        print("\n===== ĐÁNH GIÁ ENSEMBLE =====")
        
        y_true = []
        y_pred = []
        
        for seq in tqdm(sequences, desc="Evaluating"):
            pred, _ = self.predict(seq['features'])
            y_true.append(seq['label'])
            y_pred.append(pred)
        
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        
        print(f"\nEnsemble Accuracy: {accuracy:.2%}")
        
        print("\n--- Classification Report ---")
        print(classification_report(
            y_true, y_pred,
            target_names=['Non-Violent', 'Violent']
        ))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_true, y_pred)
        print(f"                Predicted")
        print(f"              Non-V  Violent")
        print(f"Actual Non-V  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Violent{cm[1][0]:5d}  {cm[1][1]:5d}")
        
        return {
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TemporalTransformer(nn.Module):
    """Transformer để học temporal dependencies"""
    
    def __init__(self, input_size=25, d_model=128, nhead=8, 
                 num_layers=4, num_classes=2, dropout=0.3):
        super(TemporalTransformer, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, seq_len=15, features=25)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

class TransformerClassifier:
    """Wrapper cho Transformer model"""
    
    def __init__(self, input_size=25, d_model=128, nhead=8, 
                 num_layers=4, dropout=0.3, device=None):
        if device is None:
            self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Sử dụng device: {self.device}")
        
        self.model = TemporalTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout
        ).to(self.device)
        
        self.is_trained = False
    
    def train(self, sequences, test_size=0.2, batch_size=32, 
              epochs=100, lr=0.0001, weight_decay=1e-4):
        """Train Transformer model"""
        print("\n===== CHUẨN BỊ DỮ LIỆU TRANSFORMER =====")
        
        labels = [seq['label'] for seq in sequences]
        train_seqs, test_seqs = train_test_split(
            sequences, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = FlowSequenceDataset(train_seqs)
        test_dataset = FlowSequenceDataset(test_seqs)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Class weights
        train_labels = train_dataset.labels.numpy()
        class_counts = np.bincount(train_labels)
        class_weights = torch.FloatTensor(
            [len(train_labels) / (2 * c) for c in class_counts]
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        print("\n===== TRAINING TRANSFORMER =====")
        best_test_acc = 0.0
        train_losses = []
        test_accs = []
        start_time = time.time()
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Evaluation
            self.model.eval()
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            test_acc = 100 * test_correct / test_total
            scheduler.step(test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), 'best_transformer_model.pth')
            
            train_losses.append(avg_train_loss)
            test_accs.append(test_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {avg_train_loss:.4f} "
                      f"Train Acc: {train_acc:.2f}% "
                      f"Test Acc: {test_acc:.2f}%")
        elapsed = time.time() - start_time
        print(f"\Training time CNN-LSTM: {elapsed/60:.1f} minutes")
        print(f"Average time per epoch: {elapsed/epochs:.1f} seconds")
        self.model.load_state_dict(torch.load('best_transformer_model.pth'))
        
        print("\n===== ĐÁNH GIÁ CUỐI CÙNG =====")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")
        
        print("\n--- Classification Report ---")
        print(classification_report(
            all_labels, all_preds,
            target_names=['Non-Violent', 'Violent']
        ))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(all_labels, all_preds)
        print(f"                Predicted")
        print(f"              Non-V  Violent")
        print(f"Actual Non-V  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Violent{cm[1][0]:5d}  {cm[1][1]:5d}")
        
        self.is_trained = True
        
        return {
            'best_test_accuracy': best_test_acc,
            'train_losses': train_losses,
            'test_accuracies': test_accs
        }
    
    def predict(self, sequence):
        """Dự đoán từ sequence"""
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        
        self.model.eval()
        
        dataset = FlowSequenceDataset([{'features': sequence, 'label': 0}])
        x = dataset.data[0].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
        
        return prediction, probabilities
    
    def save_model(self, model_path='transformer_classifier.pth'):
        """Lưu model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'is_trained': self.is_trained
        }, model_path)
        print(f"Đã lưu Transformer model vào: {model_path}")
    
    def load_model(self, model_path='transformer_classifier.pth'):
        """Load model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        self.model.eval()
        print(f"Đã load Transformer model từ: {model_path}")

class BiGRU(nn.Module):
    """Bidirectional GRU - nhẹ hơn LSTM"""
    
    def __init__(self, input_size=25, hidden_size=128, num_layers=3,
                 num_classes=2, dropout=0.4):
        super(BiGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 15, 25)
        gru_out, _ = self.gru(x)  # (batch, 15, hidden*2)
        
        # Attention weights
        attn_weights = F.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        
        output = self.fc(context)
        return output

class GRUClassifier:
    """Wrapper cho BiGRU model"""
    
    def __init__(self, input_size=25, hidden_size=128, num_layers=3,
                 dropout=0.4, device=None):
        if device is None:
            self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Sử dụng device: {self.device}")
        
        self.model = BiGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout
        ).to(self.device)
        
        self.is_trained = False
    
    def train(self, sequences, test_size=0.2, batch_size=32, 
              epochs=100, lr=0.001, weight_decay=1e-4):
        """Train GRU model"""
        print("\n===== CHUẨN BỊ DỮ LIỆU GRU =====")
        
        labels = [seq['label'] for seq in sequences]
        train_seqs, test_seqs = train_test_split(
            sequences, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = FlowSequenceDataset(train_seqs)
        test_dataset = FlowSequenceDataset(test_seqs)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Class weights
        train_labels = train_dataset.labels.numpy()
        class_counts = np.bincount(train_labels)
        class_weights = torch.FloatTensor(
            [len(train_labels) / (2 * c) for c in class_counts]
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        print("\n===== TRAINING GRU =====")
        best_test_acc = 0.0
        train_losses = []
        test_accs = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Evaluation
            self.model.eval()
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            test_acc = 100 * test_correct / test_total
            scheduler.step(test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), 'best_gru_model.pth')
            
            train_losses.append(avg_train_loss)
            test_accs.append(test_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {avg_train_loss:.4f} "
                      f"Train Acc: {train_acc:.2f}% "
                      f"Test Acc: {test_acc:.2f}%")
        
        self.model.load_state_dict(torch.load('best_gru_model.pth'))
        
        print("\n===== ĐÁNH GIÁ CUỐI CÙNG =====")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")
        
        print("\n--- Classification Report ---")
        print(classification_report(
            all_labels, all_preds,
            target_names=['Non-Violent', 'Violent']
        ))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(all_labels, all_preds)
        print(f"                Predicted")
        print(f"              Non-V  Violent")
        print(f"Actual Non-V  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Violent{cm[1][0]:5d}  {cm[1][1]:5d}")
        
        self.is_trained = True
        
        return {
            'best_test_accuracy': best_test_acc,
            'train_losses': train_losses,
            'test_accuracies': test_accs
        }
    
    def predict(self, sequence):
        """Dự đoán từ sequence"""
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        
        self.model.eval()
        
        dataset = FlowSequenceDataset([{'features': sequence, 'label': 0}])
        x = dataset.data[0].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
        
        return prediction, probabilities
    
    def save_model(self, model_path='gru_classifier.pth'):
        """Lưu model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'is_trained': self.is_trained
        }, model_path)
        print(f"Đã lưu GRU model vào: {model_path}")
    
    def load_model(self, model_path='gru_classifier.pth'):
        """Load model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        self.model.eval()
        print(f"Đã load GRU model từ: {model_path}")

class SVMClassifier:
    """SVM với RBF kernel"""
    
    def __init__(self):
        self.model = SVC(
            kernel='linear',
            C=10.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42,
            verbose=True # Thêm dòng này để thấy log quá trình chạy
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def sequence_to_feature_vector(self, sequence):
        features = []
        
        for frame_features in sequence:
            features.extend([
                frame_features['mean_magnitude'],
                frame_features['std_magnitude'],
                frame_features['max_magnitude'],
                frame_features['min_magnitude'],
                frame_features['mean_angle'],
                frame_features['std_angle'],
                frame_features['mag_percentile_25'],
                frame_features['mag_percentile_50'],
                frame_features['mag_percentile_75'],
            ])
            features.extend(frame_features['mag_hist'].tolist())
            features.extend(frame_features['ang_hist'].tolist())
        
        return np.array(features)
    
    def prepare_data(self, sequences):
        X = []
        y = []
        
        for seq in sequences:
            feature_vector = self.sequence_to_feature_vector(seq['features'])
            X.append(feature_vector)
            y.append(seq['label'])
        
        return np.array(X), np.array(y)
    
    def train(self, sequences, test_size=0.2):
        print("\n===== CHUẨN BỊ DỮ LIỆU SVM =====")
        X, y = self.prepare_data(sequences)
        
        print(f"Số lượng samples: {len(X)}")
        print(f"Số chiều features: {X.shape[1]}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        print("\n===== CHUẨN HÓA DỮ LIỆU =====")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n===== TRAINING SVM =====")
        self.model.fit(X_train_scaled, y_train)
        
        print("\n===== ĐÁNH GIÁ MODEL =====")
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        y_pred = self.model.predict(X_test_scaled)
        
        print("\n--- Classification Report ---")
        print(classification_report(
            y_test, y_pred,
            target_names=['Non-Violent', 'Violent']
        ))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                Predicted")
        print(f"              Non-V  Violent")
        print(f"Actual Non-V  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Violent{cm[1][0]:5d}  {cm[1][1]:5d}")
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, sequence):
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        
        X = self.sequence_to_feature_vector(sequence).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        return prediction, probabilities
    
    def save_model(self, model_path='svm_classifier.pkl'):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }, model_path)
        print(f"Đã lưu SVM model vào: {model_path}")
    
    def load_model(self, model_path='svm_classifier.pkl'):
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']
        print(f"Đã load SVM model từ: {model_path}")

# Import XGBoost nếu có
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

class XGBoostClassifier:
    """XGBoost classifier"""
    
    def __init__(self):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost không được cài đặt. Chạy: pip install xgboost")
        
        # Kiểm tra version XGBoost
        import xgboost
        xgb_version = tuple(map(int, xgboost.__version__.split('.')[:2]))
        
        if xgb_version >= (2, 0):
            # XGBoost >= 2.0: early_stopping_rounds trong __init__
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                early_stopping_rounds=20  # Thêm vào đây
            )
        else:
            # XGBoost < 2.0: early_stopping_rounds trong fit()
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.xgb_version = xgb_version
    
    def sequence_to_feature_vector(self, sequence):
        features = []
        
        for frame_features in sequence:
            features.extend([
                frame_features['mean_magnitude'],
                frame_features['std_magnitude'],
                frame_features['max_magnitude'],
                frame_features['min_magnitude'],
                frame_features['mean_angle'],
                frame_features['std_angle'],
                frame_features['mag_percentile_25'],
                frame_features['mag_percentile_50'],
                frame_features['mag_percentile_75'],
            ])
            features.extend(frame_features['mag_hist'].tolist())
            features.extend(frame_features['ang_hist'].tolist())
        
        return np.array(features)
    
    def prepare_data(self, sequences):
        X = []
        y = []
        
        for seq in sequences:
            feature_vector = self.sequence_to_feature_vector(seq['features'])
            X.append(feature_vector)
            y.append(seq['label'])
        
        return np.array(X), np.array(y)
    
    def train(self, sequences, test_size=0.2):
        print("\n===== CHUẨN BỊ DỮ LIỆU XGBOOST =====")
        X, y = self.prepare_data(sequences)
        
        print(f"Số lượng samples: {len(X)}")
        print(f"Số chiều features: {X.shape[1]}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        print("\n===== CHUẨN HÓA DỮ LIỆU =====")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n===== TRAINING XGBOOST =====")
        
        if self.xgb_version >= (2, 0):
            # XGBoost >= 2.0: early_stopping_rounds đã ở __init__
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=True
            )
        else:
            # XGBoost < 2.0: early_stopping_rounds ở fit()
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                early_stopping_rounds=20,
                verbose=True
            )
        
        print("\n===== ĐÁNH GIÁ MODEL =====")
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        y_pred = self.model.predict(X_test_scaled)
        
        print("\n--- Classification Report ---")
        print(classification_report(
            y_test, y_pred,
            target_names=['Non-Violent', 'Violent']
        ))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                Predicted")
        print(f"              Non-V  Violent")
        print(f"Actual Non-V  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Violent{cm[1][0]:5d}  {cm[1][1]:5d}")
        
        print("\n--- Top 10 Important Features ---")
        importance = self.model.feature_importances_
        top_indices = np.argsort(importance)[-10:][::-1]
        for idx in top_indices:
            print(f"Feature {idx}: {importance[idx]:.4f}")
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, sequence):
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        
        X = self.sequence_to_feature_vector(sequence).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        return prediction, probabilities
    
    def save_model(self, model_path='xgboost_classifier.pkl'):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }, model_path)
        print(f"Đã lưu XGBoost model vào: {model_path}")
    
    def load_model(self, model_path='xgboost_classifier.pkl'):
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']
        print(f"Đã load XGBoost model từ: {model_path}")

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, dilation, dropout=0.2):
        super().__init__()
        
        # Causal padding để output size = input size
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
                          if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """Crop padding từ output của conv1d để có causal convolution"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x

class TCN(nn.Module):
    """Temporal Convolutional Network"""
    
    def __init__(self, input_size=25, num_channels=[64, 128, 256], 
                 kernel_size=3, dropout=0.3, num_classes=2):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation = 2 ** i
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation, dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.network(x)     # (batch, channels, seq_len) - same seq_len
        x = x.mean(dim=2)       # Global average pooling -> (batch, channels)
        x = self.fc(x)          # (batch, num_classes)
        return x

class TCNClassifier:
    """Wrapper cho TCN model"""
    
    def __init__(self, input_size=25, num_channels=[64, 128, 256],
                 kernel_size=3, dropout=0.3, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Sử dụng device: {self.device}")
        
        self.model = TCN(
            input_size=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            num_classes=2
        ).to(self.device)
        
        self.is_trained = False
    
    def train(self, sequences, test_size=0.2, batch_size=32, 
              epochs=100, lr=0.001, weight_decay=1e-4):
        print("\n===== CHUẨN BỊ DỮ LIỆU TCN =====")
        
        labels = [seq['label'] for seq in sequences]
        train_seqs, test_seqs = train_test_split(
            sequences, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = FlowSequenceDataset(train_seqs)
        test_dataset = FlowSequenceDataset(test_seqs)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        train_labels = train_dataset.labels.numpy()
        class_counts = np.bincount(train_labels)
        class_weights = torch.FloatTensor(
            [len(train_labels) / (2 * c) for c in class_counts]
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        print("\n===== TRAINING TCN =====")
        best_test_acc = 0.0
        train_losses = []
        test_accs = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            self.model.eval()
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            test_acc = 100 * test_correct / test_total
            scheduler.step(test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), 'best_tcn_model.pth')
            
            train_losses.append(avg_train_loss)
            test_accs.append(test_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {avg_train_loss:.4f} "
                      f"Train Acc: {train_acc:.2f}% "
                      f"Test Acc: {test_acc:.2f}%")
        
        self.model.load_state_dict(torch.load('best_tcn_model.pth'))
        
        print("\n===== ĐÁNH GIÁ CUỐI CÙNG =====")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")
        
        print("\n--- Classification Report ---")
        print(classification_report(
            all_labels, all_preds,
            target_names=['Non-Violent', 'Violent']
        ))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(all_labels, all_preds)
        print(f"                Predicted")
        print(f"              Non-V  Violent")
        print(f"Actual Non-V  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Violent{cm[1][0]:5d}  {cm[1][1]:5d}")
        
        self.is_trained = True
        
        return {
            'best_test_accuracy': best_test_acc,
            'train_losses': train_losses,
            'test_accuracies': test_accs
        }
    
    def predict(self, sequence):
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        
        self.model.eval()
        
        dataset = FlowSequenceDataset([{'features': sequence, 'label': 0}])
        x = dataset.data[0].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
        
        return prediction, probabilities
    
    def save_model(self, model_path='tcn_classifier.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'is_trained': self.is_trained
        }, model_path)
        print(f"Đã lưu TCN model vào: {model_path}")
    
    def load_model(self, model_path='tcn_classifier.pth'):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        self.model.eval()
        print(f"Đã load TCN model từ: {model_path}")

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        self.shortcut = nn.Linear(in_features, out_features) \
                        if in_features != out_features else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += identity
        out = F.relu(out)
        return out

class ResNetLSTM(nn.Module):
    """Kết hợp Residual connections với LSTM"""
    
    def __init__(self, input_size=25, hidden_size=128, 
                 num_layers=2, num_classes=2, dropout=0.4):
        super(ResNetLSTM, self).__init__()
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(input_size if i == 0 else 128, 128, dropout)
            for i in range(3)
        ])
        
        self.lstm = nn.LSTM(
            128, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        x = x.view(batch_size * seq_len, features)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = x.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        
        x = self.fc(x)
        return x

class ResNetLSTMClassifier:
    """Wrapper cho ResNet-LSTM model"""
    
    def __init__(self, input_size=25, hidden_size=128, num_layers=2,
                 dropout=0.4, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Sử dụng device: {self.device}")
        
        self.model = ResNetLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout
        ).to(self.device)
        
        self.is_trained = False
    
    def train(self, sequences, test_size=0.2, batch_size=32, 
              epochs=100, lr=0.001, weight_decay=1e-4):
        print("\n===== CHUẨN BỊ DỮ LIỆU RESNET-LSTM =====")
        
        labels = [seq['label'] for seq in sequences]
        train_seqs, test_seqs = train_test_split(
            sequences, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = FlowSequenceDataset(train_seqs)
        test_dataset = FlowSequenceDataset(test_seqs)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        train_labels = train_dataset.labels.numpy()
        class_counts = np.bincount(train_labels)
        class_weights = torch.FloatTensor(
            [len(train_labels) / (2 * c) for c in class_counts]
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        print("\n===== TRAINING RESNET-LSTM =====")
        best_test_acc = 0.0
        train_losses = []
        test_accs = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            self.model.eval()
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            test_acc = 100 * test_correct / test_total
            scheduler.step(test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), 'best_resnet_lstm_model.pth')
            
            train_losses.append(avg_train_loss)
            test_accs.append(test_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {avg_train_loss:.4f} "
                      f"Train Acc: {train_acc:.2f}% "
                      f"Test Acc: {test_acc:.2f}%")
        
        self.model.load_state_dict(torch.load('best_resnet_lstm_model.pth'))
        
        print("\n===== ĐÁNH GIÁ CUỐI CÙNG =====")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")
        
        print("\n--- Classification Report ---")
        print(classification_report(
            all_labels, all_preds,
            target_names=['Non-Violent', 'Violent']
        ))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(all_labels, all_preds)
        print(f"                Predicted")
        print(f"              Non-V  Violent")
        print(f"Actual Non-V  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Violent{cm[1][0]:5d}  {cm[1][1]:5d}")
        
        self.is_trained = True
        
        return {
            'best_test_accuracy': best_test_acc,
            'train_losses': train_losses,
            'test_accuracies': test_accs
        }
    
    def predict(self, sequence):
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        
        self.model.eval()
        
        dataset = FlowSequenceDataset([{'features': sequence, 'label': 0}])
        x = dataset.data[0].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
        
        return prediction, probabilities
    
    def save_model(self, model_path='resnet_lstm_classifier.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'is_trained': self.is_trained
        }, model_path)
        print(f"Đã lưu ResNet-LSTM model vào: {model_path}")
    
    def load_model(self, model_path='resnet_lstm_classifier.pth'):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        self.model.eval()
        print(f"Đã load ResNet-LSTM model từ: {model_path}")

class VideoClassifier:
    """Phân loại hành động trong video mới"""
    
    def __init__(self, model_path=None, model_type='rf', ensemble_models=None):
        """
        Args:
            model_path: Đường dẫn model (nếu dùng single model)
            model_type: 'rf', 'cnn_lstm', '3dcnn', hoặc 'ensemble'
            ensemble_models: Dict of models nếu dùng ensemble
        """
        self.extractor = FlowExtractor(flow_sequence_length=15)
        self.model_type = model_type
        
        if model_type == 'ensemble':
            if ensemble_models is None:
                raise ValueError("Phải cung cấp ensemble_models khi dùng ensemble")
            self.classifier = ensemble_models
        else:
            if model_type == 'rf':
                self.classifier = ActionClassifier()
                self.classifier.load_model(model_path)
            elif model_type == 'cnn_lstm':
                self.classifier = CNNLSTMClassifier()
                self.classifier.load_model(model_path)
            elif model_type == '3dcnn':
                self.classifier = CNN3DClassifier()
                self.classifier.load_model(model_path)
            else:
                raise ValueError("model_type phải là 'rf', 'cnn_lstm', '3dcnn', hoặc 'ensemble'")
        
    def classify_video(self, video_path, output_video=None):
        """
        Phân loại hành động trong video
        
        Args:
            video_path: Đường dẫn video
            output_video: Đường dẫn video output (có annotation)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            return
        
        # Setup video writer
        if output_video:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        person_flows = defaultdict(lambda: deque(maxlen=15))
        person_predictions = {}
        prev_gray = None
        
        frame_count = 0
        
        print(f"\nPhân loại video: {video_path}")
        print(f"Model: {self.model_type.upper()}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            results = self.extractor.model.track(
                frame, persist=True, classes=[0], verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    if prev_gray is not None:
                        flow = self.extractor.calculate_optical_flow(
                            prev_gray, curr_gray, box
                        )
                        
                        if flow is not None:
                            features = self.extractor.extract_flow_features(flow)
                            
                            if features is not None:
                                person_flows[track_id].append(features)
                                
                                # Nếu đủ 15 frames, dự đoán
                                if len(person_flows[track_id]) == 15:
                                    sequence = list(person_flows[track_id])
                                    pred, probs = self.classifier.predict(sequence)
                                    person_predictions[track_id] = {
                                        'label': pred,
                                        'probs': probs
                                    }
                    
                    # Vẽ annotation
                    if track_id in person_predictions:
                        pred_info = person_predictions[track_id]
                        label_text = "VIOLENT" if pred_info['label'] == 1 else "NON-VIOLENT"
                        confidence = pred_info['probs'][pred_info['label']]
                        
                        color = (0, 0, 255) if pred_info['label'] == 1 else (0, 255, 0)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID:{track_id} {label_text}",
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, color, 2)
                        cv2.putText(frame, f"Conf: {confidence:.2f}",
                                  (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.4, color, 1)
                    else:
                        # Chưa đủ dữ liệu
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, f"ID:{track_id} Analyzing...",
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (255, 255, 0), 2)
            
            if output_video:
                out.write(frame)
            
            prev_gray = curr_gray.copy()
        
        cap.release()
        if output_video:
            out.release()
            print(f"Đã lưu video kết quả: {output_video}")
        
        # In kết quả
        print("\n===== KẾT QUẢ PHÂN LOẠI =====")
        for person_id, pred_info in person_predictions.items():
            label_text = "VIOLENT" if pred_info['label'] == 1 else "NON-VIOLENT"
            print(f"Person {person_id}: {label_text} "
                  f"(confidence: {pred_info['probs'][pred_info['label']]:.2f})")

def load_data(file_path='dataset.pkl'):
    print(f"Đang load dataset từ {file_path}...")
    with open(file_path, 'rb') as f:
        all_sequences = pickle.load(f)
    return all_sequences


if __name__ == "__main__":
    step = 2
    model_rf = False
    model_cnn_lstm = False
    model_cnn3d = False
    model_transformer = False
    model_gru = False
    model_tcn = False
    model_resnet_lstm = False
    model_svm = False
    model_xgboost = False
    model_ensemble = True
    all_model = False
    test = False
    # ===== BƯỚC 1: XÂY DỰNG DATASET =====
    print("=" * 60)
    print("BƯỚC 1: XÂY DỰNG DATASET TỪ VIDEOS")
    print("=" * 60)
    start_time = time.time()
    if step == 1:
        builder = DatasetBuilder(data_dir='data', flow_sequence_length=15)
        sequences = builder.build_dataset(output_path='dataset.pkl')
    if step >= 2:
        sequences = load_data()
        labels = [seq['label'] for seq in sequences]
        train_sequences, test_sequences = train_test_split(
            sequences, test_size=0.2, random_state=42, stratify=labels
        )
        if model_svm:
            # ===== BƯỚC 2I: TRAIN SVM =====
            print("\n" + "=" * 60)
            print("BƯỚC 2I: TRAIN SVM")
            print("=" * 60)
            start_time = time.time()
            svm_classifier = SVMClassifier()
            svm_results = svm_classifier.train(
                sequences,
                test_size=0.2
            )
            elapsed = time.time() - start_time
            print(f"\Training time SVM: {elapsed/60:.1f} minutes")
            print(f"Average time per epoch: {elapsed/epochs:.1f} seconds")
            svm_classifier.save_model('svm_classifier.pkl')
        if model_resnet_lstm:
            # ===== BƯỚC 2H: TRAIN RESNET-LSTM =====
            print("\n" + "=" * 60)
            print("BƯỚC 2H: TRAIN RESNET-LSTM")
            print("=" * 60)
            device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
            start_time = time.time()
            resnet_lstm_classifier = ResNetLSTMClassifier(
                input_size=25,
                hidden_size=128,
                num_layers=2,
                dropout=0.4,
                device=device
            )
            resnet_lstm_results = resnet_lstm_classifier.train(
                sequences,
                test_size=0.2,
                batch_size=32,
                epochs=100,
                lr=0.001
            )
            elapsed = time.time() - start_time
            print(f"\Training time ResNet-LSTM: {elapsed/60:.1f} minutes")
            print(f"Average time per epoch: {elapsed/epochs:.1f} seconds")
            resnet_lstm_classifier.save_model('resnet_lstm_classifier.pth')
        if model_tcn:
            # ===== BƯỚC 2G: TRAIN TCN =====
            print("\n" + "=" * 60)
            print("BƯỚC 2G: TRAIN TCN")
            print("=" * 60)
            device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
            start_time = time.time()
            tcn_classifier = TCNClassifier(
                input_size=25,
                num_channels=[64, 128, 256],
                kernel_size=3,
                dropout=0.3,
                device=device
            )
            tcn_results = tcn_classifier.train(
                sequences,
                test_size=0.2,
                batch_size=32,
                epochs=100,
                lr=0.001
            )
            elapsed = time.time() - start_time
            print(f"\Training time TCN: {elapsed/60:.1f} minutes")
            print(f"Average time per epoch: {elapsed/epochs:.1f} seconds")
            tcn_classifier.save_model('tcn_classifier.pth')
        if model_xgboost:
            # ===== BƯỚC 2F: TRAIN XGBOOST =====
            print("\n" + "=" * 60)
            print("BƯỚC 2F: TRAIN XGBOOST")
            print("=" * 60)
            start_time = time.time()
            xgb_classifier = XGBoostClassifier()
            xgb_results = xgb_classifier.train(
                sequences,
                test_size=0.2
            )
            elapsed = time.time() - start_time
            print(f"\Training time XGBoost: {elapsed/60:.1f} minutes")
            print(f"Average time per epoch: {elapsed/epochs:.1f} seconds")
            xgb_classifier.save_model('xgboost_classifier.pkl')
        if model_gru:
            # ===== BƯỚC 2E: TRAIN GRU =====
            print("\n" + "=" * 60)
            print("BƯỚC 2E: TRAIN GRU")
            print("=" * 60)
            device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
            start_time = time.time()
            gru_classifier = GRUClassifier(
                input_size=25,
                hidden_size=128,
                num_layers=3,
                dropout=0.4,
                device=device
            )
            gru_results = gru_classifier.train(
                sequences,
                test_size=0.2,
                batch_size=32,
                epochs=100,
                lr=0.001
            )
            elapsed = time.time() - start_time
            print(f"\Training time GRU: {elapsed/60:.1f} minutes")
            print(f"Average time per epoch: {elapsed/epochs:.1f} seconds")
            gru_classifier.save_model('gru_classifier.pth')
        if model_transformer:
            # ===== BƯỚC 2D: TRAIN TRANSFORMER =====
            print("\n" + "=" * 60)
            print("BƯỚC 2D: TRAIN TRANSFORMER")
            print("=" * 60)
            device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
            start_time = time.time()
            transformer_classifier = TransformerClassifier(
                input_size=25,
                d_model=128,
                nhead=8,
                num_layers=4,
                dropout=0.3,
                device=device
            )
            transformer_results = transformer_classifier.train(
                sequences,
                test_size=0.2,
                batch_size=32,
                epochs=100,
                lr=0.0001
            )
            elapsed = time.time() - start_time
            print(f"\Training time Transformer: {elapsed/60:.1f} minutes")
            print(f"Average time per epoch: {elapsed/epochs:.1f} seconds")
            transformer_classifier.save_model('transformer_classifier.pth')
        if model_rf:
            # ===== BƯỚC 2A: TRAIN RANDOM FOREST =====
            print("\n" + "=" * 60)
            print("BƯỚC 2A: TRAIN RANDOM FOREST")
            print("=" * 60)
            start_time = time.time()
            rf_classifier = RandomForestClassifier()
            rf_results = rf_classifier.train(train_sequences, test_size=0.25)  # 0.25 của train = 0.2 tổng
            elapsed = time.time() - start_time
            print(f"\Training time RF: {elapsed/60:.1f} minutes")
            print(f"Average time per epoch: {elapsed/epochs:.1f} seconds")
            rf_classifier.save_model('rf_classifier.pkl')
        if model_cnn_lstm:
            # ===== BƯỚC 2B: TRAIN CNN-LSTM =====
            print("\n" + "=" * 60)
            print("BƯỚC 2B: TRAIN CNN-LSTM")
            print("=" * 60)
            start_time = time.time()
            cnn_lstm_classifier = CNNLSTMClassifier(
                input_size=25,
                hidden_size=128,
                num_layers=2,
                dropout=0.5,
                device=device
            )
            cnn_lstm_results = cnn_lstm_classifier.train(
                train_sequences,
                test_size=0.25,
                batch_size=32,
                epochs=100,
                lr=0.001
            )
            elapsed = time.time() - start_time
            print(f"\Training time CNN-LSTM: {elapsed/60:.1f} minutes")
            print(f"Average time per epoch: {elapsed/epochs:.1f} seconds")
            cnn_lstm_classifier.save_model('cnn_lstm_classifier.pth')
        if model_cnn3d:
            # ===== BƯỚC 2C: TRAIN 3D CNN =====
            print("\n" + "=" * 60)
            print("BƯỚC 2C: TRAIN 3D CNN")
            print("=" * 60)
            start_time = time.time()
            cnn3d_classifier = CNN3DClassifier(dropout=0.5)
            cnn3d_results = cnn3d_classifier.train(
                train_sequences,
                test_size=0.25,
                batch_size=32,
                epochs=100,
                lr=0.001
            )
            elapsed = time.time() - start_time
            print(f"\Training time 3D CNN: {elapsed/60:.1f} minutes")
            print(f"Average time per epoch: {elapsed/epochs:.1f} seconds")
            cnn3d_classifier.save_model('3dcnn_classifier.pth')
        if model_ensemble:
            # ===== BƯỚC 3: TẠO ENSEMBLE =====
            print("\n" + "=" * 60)
            print("BƯỚC 3: TẠO VÀ ĐÁNH GIÁ ENSEMBLE")
            print("=" * 60)
            
            # Load lại các models
            # rf = RandomForestClassifier()
            # rf.load_model('rf_classifier.pkl')
            
            # cnn_lstm = CNNLSTMClassifier()
            # cnn_lstm.load_model('cnn_lstm_classifier.pth')
            
            # cnn3d = CNN3DClassifier()
            # cnn3d.load_model('3dcnn_classifier.pth')

            xg = XGBoostClassifier()
            xg.load_model('xgboost_classifier.pkl')
            tcn = TCNClassifier()
            tcn.load_model('tcn_classifier.pth')
            tran = TransformerClassifier()
            tran.load_model('transformer_classifier.pth')

            
            # Tạo ensemble với weighted strategy
            # Weights dựa trên performance của từng model
            ensemble_weights = {
                # 'rf': rf_results['test_accuracy'],
                # 'cnn_lstm': cnn_lstm_results['best_test_accuracy'] / 100,
                # 'cnn3d': cnn3d_results['best_test_accuracy'] / 100
                'xg': 0.3,
                'tcn': 0.3,
                'tran': 0.4
            }
            
            ensemble = EnsembleClassifier(
                models={
                    # 'rf': rf,
                    # 'cnn_lstm': cnn_lstm,
                    # 'cnn3d': cnn3d,
                    'xg': xg,
                    'tcn': tcn,
                    'tran': tran
                },
                weights=ensemble_weights,
                strategy='weighted'
            )
            
            # Đánh giá ensemble trên test set
            ensemble_results = ensemble.evaluate(test_sequences)
        if all_model:
            # ===== SO SÁNH TẤT CẢ MODELS =====
            print("\n" + "=" * 60)
            print("SO SÁNH HIỆU NĂNG TẤT CẢ MODELS")
            print("=" * 60)
            
            print(f"\n{'Model':<20} {'Test Accuracy':<15} {'Parameters':<20}")
            print("-" * 55)
            print(f"{'Random Forest':<20} {rf_results['test_accuracy']:.2%} {'~200 trees':<20}")
            print(f"{'CNN-LSTM':<20} {cnn_lstm_results['best_test_accuracy']:.2f}% {'~1M params':<20}")
            print(f"{'3D CNN':<20} {cnn3d_results['best_test_accuracy']:.2f}% {'~500K params':<20}")
            print(f"{'Ensemble (Weighted)':<20} {ensemble_results['accuracy']:.2%} {'All models':<20}")
            
            
            # Tìm best model
            all_accs = {
                'rf': rf_results['test_accuracy'] * 100,
                'cnn_lstm': cnn_lstm_results['best_test_accuracy'],
                'cnn3d': cnn3d_results['best_test_accuracy'],
                'ensemble': ensemble_results['accuracy'] * 100
            }
            best_model = max(all_accs, key=all_accs.get)
            
            print(f"\n🏆 Best Model: {best_model.upper()} ({all_accs[best_model]:.2f}%)")
    
    # ===== BƯỚC 4: TEST TRÊN VIDEO MỚI =====
    print("\n" + "=" * 60)
    print("BƯỚC 4: PHÂN LOẠI VIDEO MỚI")
    print("=" * 60)
    
    test_video = 'data_valid/3.mp4'
    if test and model_rf:
        # Test với từng model
        print("\n--- Test với Random Forest ---")
        rf_video = VideoClassifier(model_path='rf_classifier.pkl', model_type='rf')
        rf_video.classify_video(test_video, 'output_rf.mp4')
    if test and model_cnn_lstm:
        print("\n--- Test với CNN-LSTM ---")
        cnn_lstm_video = VideoClassifier(
            model_path='cnn_lstm_classifier.pth', 
            model_type='cnn_lstm'
        )
        cnn_lstm_video.classify_video(test_video, 'output_cnn_lstm.mp4')

    if test and model_cnn3d:
        print("\n--- Test với 3D CNN ---")
        cnn3d_video = VideoClassifier(
            model_path='3dcnn_classifier.pth',
            model_type='3dcnn'
        )
        cnn3d_video.classify_video(test_video, 'output_3dcnn.mp4')

    if test and model_ensemble:
        print("\n--- Test với Ensemble ---")
        ensemble_video = VideoClassifier(
            model_type='ensemble',
            ensemble_models=ensemble
        )
        ensemble_video.classify_video(test_video, 'output_ensemble.mp4')
    if test and all_model:
        # ===== SUMMARY =====
        print("\n" + "=" * 60)
        print("HOÀN THÀNH! 🎉")
        print("=" * 60)
        
        print("\n📁 Các file đã tạo:")
        print("  Dataset:")
        print("    - dataset.pkl")
        print("  Models:")
        print("    - rf_classifier.pkl")
        print("    - cnn_lstm_classifier.pth")
        print("    - 3dcnn_classifier.pth")
        print("  Videos:")
        print("    - output_rf.mp4")
        print("    - output_cnn_lstm.mp4")
        print("    - output_3dcnn.mp4")
        print("    - output_ensemble.mp4")
        
        print("\n📊 Performance Summary:")
        for name, acc in sorted(all_accs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:<15}: {acc:>6.2f}%")
        
        print("\n💡 Khuyến nghị:")
        if best_model == 'ensemble':
            print("  ✓ Ensemble cho kết quả tốt nhất, nhưng chậm hơn")
            print("  ✓ Dùng ensemble cho production nếu cần accuracy cao nhất")
        else:
            print(f"  ✓ {best_model.upper()} đã đạt performance tốt nhất")
            print(f"  ✓ Có thể dùng {best_model} thay vì ensemble để tăng tốc độ")
    print("\n🎉 Kết thúc quá trình!")
