import sys
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QListWidget, 
                             QGroupBox, QFormLayout,QSlider)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize, QMutex,pyqtSlot
def calculate_2d_horizontal_angle(p1, p2):
    """
    计算两点连线 (p1, p2) 与影像水平线的 2D 夹角 (0-180度，或 -180到180)
    p1, p2: MediaPipe Landmark 物件 (仅使用 x, y) 或 {'x': float, 'y': float} 字典

    返回角度，如果 p2 在 p1 的右边，且 p2.y < p1.y 则角度为负，反之为正。 
    我们将返回一个范围在 -180 到 180 之间的值。
    """
    # 确保输入是字典格式，或者有 x, y 属性
    x1, y1 = (p1.x, p1.y) if hasattr(p1, 'x') else (p1['x'], p1['y'])
    x2, y2 = (p2.x, p2.y) if hasattr(p2, 'x') else (p2['x'], p2['y'])

    delta_x = x2 - x1
    delta_y = y2 - y1 # MediaPipe 的 Y 轴是向下增长的

    # 使用 arctan2 可以正确处理所有象限，返回 -pi 到 pi 的弧度
    angle_radians = np.arctan2(delta_y, delta_x)
    return np.degrees(angle_radians)
def calculate_distance_3d(p1, p2):
    """计算两个 3D 关键点之间的真实空间距离"""
    p1_coords = np.array([p1.x, p1.y, p1.z])
    p2_coords = np.array([p2.x, p2.y, p2.z])
    distance = np.linalg.norm(p1_coords - p2_coords)
    return distance
# --- 姿态分析辅助函数 ---
def calculate_angle_3d(a, b, c):
    """计算三个 3D 关键点的角度 (b 为顶点)"""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    cosine_angle = np.clip(dot_product / (mag_ba * mag_bc + 1e-6), -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_angle_2d(a, b, c):
    """计算三个 2D 点的角度 (b 为顶点)"""
    a = np.array([a['x'], a['y']])
    b = np.array([b['x'], b['y']])
    c = np.array([c['x'], c['y']])
    ba = a - b
    bc = c - b
    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    cosine_angle = np.clip(dot_product / (mag_ba * mag_bc + 1e-6), -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_vector_angle_3d(v1_start, v1_end, v2_start, v2_end):
    """计算两个 3D 向量之间的角度"""
    v1 = np.array([v1_end.x - v1_start.x, v1_end.y - v1_start.y, v1_end.z - v1_start.z])
    v2 = np.array([v2_end.x - v2_start.x, v2_end.y - v2_start.y, v2_end.z - v2_start.z])
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def get_midpoint(p1, p2):
    """计算两个 3D Landmark 的 2D 中点 (x, y)"""
    return {'x': (p1.x + p2.x) / 2, 'y': (p1.y + p2.y) / 2}

# --- 影片处理执行 ---
class VideoThread(QThread):
    #修正：使用 pyqtSignal 定义信号
    changePixmap = pyqtSignal(QImage)
    updateStats = pyqtSignal(dict)

    def __init__(self, video_source, target_width=1280, target_height=720): # 新增参数
            super().__init__()
            self.video_source = video_source
            self.running = True
            self.paused = False
            self.target_width = target_width # 保存目标宽度
            self.target_height = target_height # 保存目标高度

    def pause(self):
        """暂停执行的处理回调"""
        self.paused = True

    def resume(self):
        """恢复执行的处理回调"""
        self.paused = False

    def stop(self):
        """安全停止执行回调"""
        self.running = False
        self.wait()

    def run(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        
        cap = cv2.VideoCapture(self.video_source)
        
        while self.running and cap.isOpened():
            if self.paused:
                self.msleep(10)
                continue
            
            success, image = cap.read()
            if not success:
                break

            img_h, img_w, _ = image.shape
            
            # --- 1. 复制原始影像作为左侧显示 ---
            original_image = image.copy() 
            
            # --- 2. 在右侧影像上进行 MediaPipe 处理和绘图 ---
            # 确保 MediaPipe 处理的是独立的影像副本
            processed_image = image.copy() 

            image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            results = pose.process(image_rgb)
            
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # 这是我们绘图的影像

            stats = {}
            feedback = []
            
            # 预设颜色预设颜色
            ski_color = (0, 255, 0)      # 绿色
            knee_line_color = (255, 255, 0) # 黄色
            inconsistency_color = (0, 255, 255) # 青色
            separation_color = (0, 255, 0) # 绿色

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    const = mp_pose.PoseLandmark

                    # --- 绘制预设骨架 (在 processed_image 上) ---
                    mp_drawing.draw_landmarks(
                        image_bgr, # 注意：现在在 image_bgr 上绘图
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )

                    # --- 所有计算指标的逻辑 (保持不变) ---
                    # 这里将包含所有的 knee, hip, parallel, angulation, knee_diff, dist_inconsistency, body_separation 计算
                    # 以及根据这些计算结果，设定 ski_color, knee_line_color, inconsistency_color, separation_color 的逻辑

                    # 1. 膝盖弯曲
                    left_knee_angle = calculate_angle_3d(landmarks[const.LEFT_HIP], landmarks[const.LEFT_KNEE], landmarks[const.LEFT_ANKLE])
                    right_knee_angle = calculate_angle_3d(landmarks[const.RIGHT_HIP], landmarks[const.RIGHT_KNEE], landmarks[const.RIGHT_ANKLE])
                    avg_knee_flexion = (left_knee_angle + right_knee_angle) / 2
                    stats['knee'] = avg_knee_flexion
                    if avg_knee_flexion > 160: feedback.append("提示: 膝盖弯曲不足")

                    # 2. 髖部前傾
                    left_hip_flexion = calculate_angle_3d(landmarks[const.LEFT_SHOULDER], landmarks[const.LEFT_HIP], landmarks[const.LEFT_KNEE])
                    right_hip_flexion = calculate_angle_3d(landmarks[const.RIGHT_SHOULDER], landmarks[const.RIGHT_HIP], landmarks[const.RIGHT_KNEE])
                    avg_hip_flexion = (left_hip_flexion + right_hip_flexion) / 2
                    stats['hip'] = avg_hip_flexion
                    if avg_hip_flexion > 170: feedback.append("提示: 上半身太直")

                    # 3. 雙腳方向與平行度分析 (區分 A板/V板/I板)
                    L_ANKLE = landmarks[const.LEFT_ANKLE]
                    R_ANKLE = landmarks[const.RIGHT_ANKLE]
                    L_FOOT = landmarks[const.LEFT_FOOT_INDEX]
                    R_FOOT = landmarks[const.RIGHT_FOOT_INDEX]

                    ankle_distance = abs(L_ANKLE.x - R_ANKLE.x) * img_w
                    foot_distance = abs(L_FOOT.x - R_FOOT.x) * img_w

                    foot_parallel_angle = calculate_vector_angle_3d(L_ANKLE, L_FOOT, R_ANKLE, R_FOOT)
                    stats['parallel'] = foot_parallel_angle
                    
                    PARALLEL_THRESHOLD = 20 
                    V_SHAPE_THRESHOLD = 50 

                    if foot_parallel_angle > PARALLEL_THRESHOLD:
                        if (ankle_distance - foot_distance) > V_SHAPE_THRESHOLD and (ankle_distance > 0):
                            ski_color = (255, 165, 0) # 橙色
                            feedback.append(f"姿态: 侦测到 V 板 (内八字), 距离差: {ankle_distance - foot_distance:.0f}px")
                        else:
                            ski_color = (0, 0, 255) # 红色
                            feedback.append(f"警告: 双脚不平行 (A字型), 角度差: {foot_parallel_angle:.1f}°")
                    else:
                        ski_color = (0, 255, 0) # 绿色
                        feedback.append(f"姿态: 良好 I 板 (平行), 角度差: {foot_parallel_angle:.1f}°")
                        
                    # 4. C型反弓
                    mid_shoulder = get_midpoint(landmarks[const.LEFT_SHOULDER], landmarks[const.RIGHT_SHOULDER])
                    mid_hip = get_midpoint(landmarks[const.LEFT_HIP], landmarks[const.RIGHT_HIP])
                    mid_ankle = get_midpoint(landmarks[const.LEFT_ANKLE], landmarks[const.RIGHT_ANKLE])
                    angulation_c_shape = calculate_angle_2d(mid_shoulder, mid_hip, mid_ankle)
                    stats['angulation'] = angulation_c_shape
                    if angulation_c_shape > 175: feedback.append("提示: 仅内倾 (I型), 增加反弓 (C型)")

                    # 5. 前后平衡 (Knee Over Toe - 膝盖超过脚尖)
                    left_hip_x = landmarks[const.LEFT_HIP].x
                    left_ankle_x = landmarks[const.LEFT_ANKLE].x
                    right_hip_x = landmarks[const.RIGHT_HIP].x
                    right_ankle_x = landmarks[const.RIGHT_ANKLE].x

                    left_lead_dist = left_hip_x - left_ankle_x
                    right_lead_dist = right_hip_x - right_ankle_x

                    avg_lead_dist = (left_lead_dist + right_lead_dist) / 2 * img_w # 转为像素距离
                    LEAD_THRESHOLD = -20 # 负值表示髋部落后于脚踝

                    stats['lead_dist'] = avg_lead_dist
                    if avg_lead_dist < LEAD_THRESHOLD:
                        feedback.append("警告: 重心后坐 (Hip落后于Ankle)")


                    # 6. 双膝高低差异检测 (Knee Alignment)
                    L_KNEE_Y = landmarks[const.LEFT_KNEE].y
                    R_KNEE_Y = landmarks[const.RIGHT_KNEE].y
                    knee_vertical_diff_px = abs(L_KNEE_Y - R_KNEE_Y) * img_h
                    KNEE_VERTICAL_THRESHOLD = 50 

                    stats['knee_diff'] = knee_vertical_diff_px
                    if knee_vertical_diff_px > KNEE_VERTICAL_THRESHOLD:
                        feedback.append(f"警告: 双膝高低差异过大 ({knee_vertical_diff_px:.0f}px)")
                        knee_line_color = (0, 165, 255) # 橘红色
                    else:
                        knee_line_color = (255, 255, 0) # 黃色

                    # 7. 双膝/双踝距离一致性检测 (Stance Consistency) - 修正使用 3D 距离
                    L_KNEE = landmarks[const.LEFT_KNEE]
                    R_KNEE = landmarks[const.RIGHT_KNEE]
                    L_ANKLE = landmarks[const.LEFT_ANKLE]
                    R_ANKLE = landmarks[const.RIGHT_ANKLE]

                    knee_3d_dist = calculate_distance_3d(L_KNEE, R_KNEE)
                    ankle_3d_dist = calculate_distance_3d(L_ANKLE, R_ANKLE)

                    dist_inconsistency_norm = abs(knee_3d_dist - ankle_3d_dist)
                    INCONSISTENCY_THRESHOLD_3D = 0.004

                    stats['dist_inconsistency'] = dist_inconsistency_norm

                    if dist_inconsistency_norm > INCONSISTENCY_THRESHOLD_3D:
                        if knee_3d_dist < ankle_3d_dist:
                            feedback.append(f"警告: 双膝内扣 (X 型倾向), 3D 差异: {dist_inconsistency_norm:.3f}")
                            inconsistency_color = (255, 0, 255) # 品红色
                        else:
                            feedback.append(f"警告: 双膝外张 (O 型倾向), 3D 差异: {dist_inconsistency_norm:.3f}")
                            inconsistency_color = (255, 0, 255) # 品红色
                    else:
                        feedback.append(f"姿态: 膝踝距离一致性良好")
                        inconsistency_color = (0, 255, 255) # 青色

                    # 8. 身体分离度检测 (Body Separation / Torsion)
                    L_SHOULDER = landmarks[const.LEFT_SHOULDER]
                    R_SHOULDER = landmarks[const.RIGHT_SHOULDER]
                    L_HIP = landmarks[const.LEFT_HIP]
                    R_HIP = landmarks[const.RIGHT_HIP]

                    shoulder_angle_2d = calculate_2d_horizontal_angle(L_SHOULDER, R_SHOULDER)
                    hip_angle_2d = calculate_2d_horizontal_angle(L_HIP, R_HIP)
                    body_separation_diff = abs(shoulder_angle_2d - hip_angle_2d)
                    SEPARATION_THRESHOLD = 15 

                    stats['body_separation'] = body_separation_diff

                    if body_separation_diff < SEPARATION_THRESHOLD:
                        feedback.append(f"警告: 身体分离度不足 (肩髋角度差: {body_separation_diff:.1f}°)！")
                        separation_color = (0, 0, 255) # 红色
                    else:
                        feedback.append(f"姿态: 身体分离度良好 ({body_separation_diff:.1f}°)！")
                        separation_color = (0, 255, 0) # 绿色

                    # --- 绘制所有自定义标记 (在 image_bgr 上) ---
                    def get_coords(landmark_id):
                        lm = landmarks[landmark_id]
                        return (int(lm.x * img_w), int(lm.y * img_h))

                    # 雪板线 (脚踝到脚尖)
                    left_ankle_pt = get_coords(const.LEFT_ANKLE)
                    left_foot_pt = get_coords(const.LEFT_FOOT_INDEX)
                    right_ankle_pt = get_coords(const.RIGHT_ANKLE)
                    right_foot_pt = get_coords(const.RIGHT_FOOT_INDEX)
                    cv2.line(image_bgr, left_ankle_pt, left_foot_pt, ski_color, 4)
                    cv2.line(image_bgr, right_ankle_pt, right_foot_pt, ski_color, 4)

                    # 双膝连线
                    left_knee_pt = get_coords(const.LEFT_KNEE)
                    right_knee_pt = get_coords(const.RIGHT_KNEE)
                    cv2.line(image_bgr, left_knee_pt, right_knee_pt, knee_line_color, 2)

                    # 膝盖-脚踝垂直连线 (显示膝踝一致性)
                    cv2.line(image_bgr, left_knee_pt, left_ankle_pt, inconsistency_color, 2)
                    cv2.line(image_bgr, right_knee_pt, right_ankle_pt, inconsistency_color, 2)

                    # 肩膀和髋部连线 (显示身体分离度)
                    left_shoulder_pt = get_coords(const.LEFT_SHOULDER)
                    right_shoulder_pt = get_coords(const.RIGHT_SHOULDER)
                    left_hip_pt = get_coords(const.LEFT_HIP)
                    right_hip_pt = get_coords(const.RIGHT_HIP)
                    cv2.line(image_bgr, left_shoulder_pt, right_shoulder_pt, separation_color, 2)
                    cv2.line(image_bgr, left_hip_pt, right_hip_pt, separation_color, 2)

                else:
                    feedback.append("未侦测到姿态")
                    
            except Exception as e:
                # print(f"分析时出错: {e}") 
                feedback.append("分析错误")

            stats['feedback'] = feedback

            # --- 将原始影像和处理后的影像水平拼接 ---
            # 确保两者高度一致，如果需要可以先resize其中一个
            if original_image.shape[0] != image_bgr.shape[0]:
                processed_image_resized = cv2.resize(image_bgr, (image_bgr.shape[1], original_image.shape[0]))
            else:
                processed_image_resized = image_bgr

            combined_image = np.hstack((original_image, processed_image_resized))

            # --- 转换拼接后的影像格式并发送信号 ---
            rgb_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放拼接后的影像以适应 GUI 视窗
            # 这里需要调整缩放比例，因为宽度变成了两倍
            scaled_image = qt_image.scaled(self.target_width, self.target_height, Qt.KeepAspectRatio) # 使用 target_width/height

            self.changePixmap.emit(scaled_image)
            self.updateStats.emit(stats)

        cap.release()
        pose.close()
        # print("影像执行结束。")

# --- 主 GUI 应用程式 ---
class SkiAnalyzerApp(QWidget):
    MAX_WINDOW_WIDTH = 1280
    MAX_WINDOW_HEIGHT = 720
    DASHBOARD_WIDTH = 300
    PADDING = 20 # 布局间隙和边框
    def __init__(self):
        super().__init__()
        self.setWindowTitle("即时滑雪姿态分析器 (Real-time Ski Pose Analyzer)")
        
        # 1. 计算留给视频显示的可用空间
        # 视频可用总宽度 = 最大窗口宽度 - 仪表板宽度 - 间隙
        video_available_total_width = self.MAX_WINDOW_WIDTH - self.DASHBOARD_WIDTH - self.PADDING
        
        # 2. 计算单边视频的最大宽度 (总宽度的一半)
        self.display_video_width = video_available_total_width // 2
        
        # 3. 设置视频的显示高度 (以保持比例，这里先设定一个合理的初始值)
        # 我们以 16:9 比例的 640x360 或 490x275 为基准，这里直接固定一个最大高度
        self.display_video_height = self.MAX_WINDOW_HEIGHT - 120 # 120px 留给按钮和标题等

        # 确保单边视频宽度合理（例如，不小于 400px）
        if self.display_video_width < 400:
             self.display_video_width = 400
             
        # 重新计算窗口总尺寸
        self.combined_width = self.display_video_width * 2
        
        self.window_total_width = self.combined_width + self.DASHBOARD_WIDTH + self.PADDING
        self.window_total_height = self.display_video_height + 120 # 120px 留给按钮和边距
        
        # 确保计算后的窗口不超过最大限制
        if self.window_total_width > self.MAX_WINDOW_WIDTH:
            self.window_total_width = self.MAX_WINDOW_WIDTH
        if self.window_total_height > self.MAX_WINDOW_HEIGHT:
            self.window_total_height = self.MAX_WINDOW_HEIGHT

        # 设定窗口大小
        self.setGeometry(100, 100, self.window_total_width, self.window_total_height) 

        self.thread = None

        # --- GUI 布局 ---
        main_layout = QHBoxLayout()

        # 左侧: 影片显示
        video_layout = QVBoxLayout()
        self.video_label = QLabel(self)
        self.video_label.setText("请选择视频或启动摄影机")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: #333;")
        
        # 这里的 minimum size 应该是组合视频的大小
        self.video_label.setMinimumSize(self.combined_width, self.display_video_height) 
        video_layout.addWidget(self.video_label)
        
        # 控制按鈕
        control_layout = QHBoxLayout()
        self.btn_open_file = QPushButton("开启检测", self)
        self.btn_open_webcam = QPushButton("启动摄影机 (Webcam)", self)
        self.btn_pause = QPushButton("暂停", self) # 新增暂停按鈕
        self.btn_stop = QPushButton("停止", self)
        
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        
        
        # 縮放控制項
        # zoom_layout = QHBoxLayout()
        # zoom_layout.addWidget(QLabel("縮放係數:"))
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(100) # 1.00x
        self.zoom_slider.setMaximum(400) # 4.00x
        self.zoom_slider.setValue(100)  # 預設 1.00x
        self.zoom_slider.setTickInterval(50)
        self.zoom_slider.setSingleStep(10)
        
        # self.lbl_zoom_value = QLabel("1.00x")
        # self.zoom_slider.valueChanged.connect(self.update_zoom_factor)

        # zoom_layout.addWidget(self.zoom_slider)
        # zoom_layout.addWidget(self.lbl_zoom_value)
        
        # control_layout.addLayout(zoom_layout) # 假設 control_layout 是您的按鈕佈局
        control_layout.addWidget(self.btn_open_file)
        control_layout.addWidget(self.btn_open_webcam)
        control_layout.addWidget(self.btn_pause) # 加入新的按鈕
        control_layout.addWidget(self.btn_stop)
        video_layout.addLayout(control_layout)



        # 右侧: 仪表板
        # 修正: 建立一个 QWidget 作为仪表板容器来设定宽度
        dashboard_container = QWidget()
        dashboard_container.setFixedWidth(300)
        
        dashboard_layout = QVBoxLayout()
        dashboard_container.setLayout(dashboard_layout)
        
        # 1. 数据显示
        stats_group = QGroupBox("姿态数据 (deg)")
        stats_layout = QFormLayout()
        self.lbl_knee = QLabel("N/A")
        self.lbl_hip = QLabel("N/A")
        self.lbl_parallel = QLabel("N/A")
        self.lbl_angulation = QLabel("N/A")
        self.lbl_lead = QLabel("N/A")
        font = QFont("Arial", 14)
        self.lbl_knee.setFont(font)
        self.lbl_hip.setFont(font)
        self.lbl_parallel.setFont(font)
        self.lbl_angulation.setFont(font)
        self.lbl_lead.setFont(font)
        # self.lbl_knee_diff = QLabel("N/A")
        # self.lbl_knee_diff.setFont(font)
        self.lbl_separation = QLabel("N/A")
        self.lbl_separation.setFont(font)
        
        self.lbl_inconsistency = QLabel("N/A")
        self.lbl_inconsistency.setFont(font)
        
        
        stats_layout.addRow(QLabel("膝盖弯曲:"), self.lbl_knee)
        stats_layout.addRow(QLabel("髋部前倾:"), self.lbl_hip)
        stats_layout.addRow(QLabel("双脚平行:"), self.lbl_parallel)
        stats_layout.addRow(QLabel("C型反弓:"), self.lbl_angulation)
        stats_layout.addRow(QLabel("前后重心 (px):"), self.lbl_lead)
        stats_layout.addRow(QLabel("膝踝距离差 (px):"), self.lbl_inconsistency)
        stats_layout.addRow(QLabel("身体分离度 (deg):"), self.lbl_separation)
        stats_group.setLayout(stats_layout)
        
        # 2. 即時建议
        feedback_group = QGroupBox("即时建议")
        feedback_layout = QVBoxLayout()
        self.feedback_list = QListWidget()
        feedback_layout.addWidget(self.feedback_list)
        feedback_group.setLayout(feedback_layout)

        dashboard_layout.addWidget(stats_group)
        dashboard_layout.addWidget(feedback_group)
        
        main_layout.addLayout(video_layout)
        main_layout.addWidget(dashboard_container) # 使用 addWidget 添加容器
        self.setLayout(main_layout)

        # --- 连接信号 ---
        self.btn_open_file.clicked.connect(self.open_file)
        self.btn_open_webcam.clicked.connect(self.open_webcam)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_pause.clicked.connect(self.toggle_pause_video) # 连接暂停/继续功能
        self.setLayout(main_layout)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.start_video_thread(file_path)

    def open_webcam(self):
        self.start_video_thread(0)

    def start_video_thread(self, source):
            self.stop_video() 
            
            # 将目标显示宽度传递给执行绪
            self.thread = VideoThread(source, 
                                  target_width=self.combined_width, 
                                  target_height=self.display_video_height)
            self.thread.changePixmap.connect(self.set_image)
            self.thread.updateStats.connect(self.update_stats)
            self.thread.finished.connect(self.on_thread_finished)
            self.thread.start()
            
            self.btn_stop.setEnabled(True)
            self.btn_pause.setEnabled(True)
            self.btn_pause.setText("暂停")
            self.btn_open_file.setEnabled(False)
            self.btn_open_webcam.setEnabled(False)

    def stop_video(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False) # 停止时禁用暂停按钮
        self.btn_pause.setText("暂停")
        self.btn_open_file.setEnabled(True)
        self.btn_open_webcam.setEnabled(True)
        self.reset_stats_display()


    def toggle_pause_video(self):
        """切换影片的暂停/继续状态"""
        if self.thread and self.thread.isRunning():
            if self.thread.paused:
                self.thread.resume()
                self.btn_pause.setText("暂停")
            else:
                self.thread.pause()
                self.btn_pause.setText("继续")


    @pyqtSlot()
    def on_thread_finished(self):
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("暂停")
        self.btn_open_file.setEnabled(True)
        self.btn_open_webcam.setEnabled(True)
        self.video_label.setText("分析结束。请选择新影片或启动摄影机。")
        self.feedback_list.clear()


    @pyqtSlot(QImage)
    def set_image(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))


    @pyqtSlot(dict)
    def update_stats(self, stats):
        # 只有在非暂停状态下才更新数据显示
        if self.thread and not self.thread.paused:
            KNEE_GOOD = 150
            HIP_GOOD = 160
            PARALLEL_BAD = 20
            ANGULATION_GOOD = 170
            LEAD_BAD = -20
            KNEE_DIFF_BAD = 50
            # INCONSISTENCY_BAD = 30
            INCONSISTENCY_BAD = 0.004
            SEPARATION_BAD_THRESHOLD = 15 # 角度差异小于这个值视为不好
            SEPARATION_GOOD_THRESHOLD = 45 # 角度差异大于这个值视为良好（但也不能无限大）
            
            self.lbl_lead.setText(f"{stats.get('lead_dist', 0):.1f}")
            self.lbl_knee.setText(f"{stats.get('knee', 0):.1f}")
            self.lbl_hip.setText(f"{stats.get('hip', 0):.1f}")
            self.lbl_parallel.setText(f"{stats.get('parallel', 0):.1f}")
            self.lbl_angulation.setText(f"{stats.get('angulation', 0):.1f}")
            # self.lbl_knee_diff.setText(f"{stats.get('knee_diff', 0):.1f}")
            # self.lbl_inconsistency.setText(f"{stats.get('dist_inconsistency', 0):.1f}")
            self.lbl_inconsistency.setText(f"{stats.get('dist_inconsistency', 0):.3f}")
            self.lbl_separation.setText(f"{stats.get('body_separation', 0):.1f}")
            
            
            self.set_label_color(self.lbl_knee, stats.get('knee', 0), KNEE_GOOD, less_is_better=True)
            self.set_label_color(self.lbl_hip, stats.get('hip', 0), HIP_GOOD, less_is_better=True)
            self.set_label_color(self.lbl_parallel, stats.get('parallel', 0), PARALLEL_BAD, is_bad_threshold=True)
            self.set_label_color(self.lbl_angulation, stats.get('angulation', 0), ANGULATION_GOOD, less_is_better=True)
            self.set_label_color(self.lbl_lead, stats.get('lead_dist', 0), LEAD_BAD, less_is_better=False)
            # self.set_label_color(self.lbl_knee_diff, stats.get('knee_diff', 0), KNEE_DIFF_BAD, less_is_better=True, is_bad_threshold=True)
            # self.set_label_color(self.lbl_inconsistency, stats.get('dist_inconsistency', 0), INCONSISTENCY_BAD, less_is_better=True, is_bad_threshold=True)
            self.set_label_color(self.lbl_inconsistency, stats.get('dist_inconsistency', 0), INCONSISTENCY_BAD, less_is_better=True, is_bad_threshold=True)
            separation_value = stats.get('body_separation', 0)
            if separation_value < SEPARATION_BAD_THRESHOLD:
                self.lbl_separation.setStyleSheet("color: #FF6B6B;") # 红色
            elif separation_value > SEPARATION_GOOD_THRESHOLD:
                self.lbl_separation.setStyleSheet("color: #6BFF6B;") # 绿色
            else:
                self.lbl_separation.setStyleSheet("color: #FFFFFF;") # 预设白色 (中等)
            self.feedback_list.clear()
            self.feedback_list.addItems(stats.get('feedback', []))

    def reset_stats_display(self):
        """重置数据显示为 N/A"""
        self.lbl_knee.setText("N/A")
        self.lbl_hip.setText("N/A")
        self.lbl_parallel.setText("N/A")
        self.lbl_angulation.setText("N/A")
        self.feedback_list.clear()
        
    def set_label_color(self, label, value, threshold, less_is_better=True, is_bad_threshold=False):
        """根据值和阈值设定标签颜色"""
        red = "color: #FF6B6B;"
        green = "color: #6BFF6B;" 
        
        if is_bad_threshold:
            if value > threshold:
                label.setStyleSheet(red)
            else:
                label.setStyleSheet(green)
        else:
            if less_is_better:
                if value < threshold:
                    label.setStyleSheet(green)
                else:
                    label.setStyleSheet(red)
            else:
                if value > threshold:
                    label.setStyleSheet(green)
                else:
                    label.setStyleSheet(red)

    def closeEvent(self, event):
        self.stop_video()
        event.accept()


# --- 执行 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SkiAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
