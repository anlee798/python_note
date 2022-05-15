import cv2
import math
import numpy as np # 数据处理的库 numpy
import mediapipe as mp
import imutils

import os
import sys
sys.path.append(r'D:\ActionDetection\DriverBehaviorDetection\YOLOv5-pytorch')
import predict


mp_face_mesh = mp.solutions.face_mesh
model = mp_face_mesh.FaceMesh(
    static_image_mode=False,       #是静态图片还是连续视频帧
    refine_landmarks=True,        #使用Attention Mesh模型，对嘴唇、眼睛、瞳孔周围的关键点精细定位
    max_num_faces=1,               #最多检测几张脸
    min_detection_confidence=0.5,  #置信度阈值，越接近1越准
    min_tracking_confidence=0.5    #追踪阈值
)
#导入可视化函数和可视化样式
mp_drawing = mp.solutions.drawing_utils
#关键点可视化样式
landmark_drawing_spec = mp_drawing.DrawingSpec(thickness = 1,circle_radius = 2,color = [66,77,229])
#轮廓可视化样式
connection_drawing_spec = mp_drawing.DrawingSpec(thickness = 2,circle_radius = 1,color = [223,155,6])

# 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
object_pts = np.float32([[6.825897, 6.760612, 4.402142],  #33左眉左上角
                         [1.330353, 7.122144, 6.903745],  #29左眉右角
                         [-1.330353, 7.122144, 6.903745], #34右眉左角
                         [-6.825897, 6.760612, 4.402142], #38右眉右上角
                         [5.311432, 5.485328, 3.987654],  #13左眼左上角
                         [1.789930, 5.393625, 4.413414],  #17左眼右上角
                         [-1.789930, 5.393625, 4.413414], #25右眼左上角
                         [-5.311432, 5.485328, 3.987654], #21右眼右上角
                         [2.005628, 1.409845, 6.165652],  #55鼻子左上角
                         [-2.005628, 1.409845, 6.165652], #49鼻子右上角
                         [2.774015, -2.080775, 5.048531], #43嘴左上角
                         [-2.774015, -2.080775, 5.048531],#39嘴右上角
                         [0.000000, -3.116408, 6.097667], #45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])#6下巴角

# 相机坐标系(XYZ)：添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]# 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

# 像素坐标系(xy)：填写凸轮的本征和畸变系数
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
# 绘制正方体12轴
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

def get_head_pose(face_landmarks,width,height):# 头部姿态估计
    # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
    # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
    # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    dics = [70,107,336,300,33,133,362,263,102,331,62,306,17,199]
    pointarr = []
    for dic in dics:
        cx = int(face_landmarks.landmark[dic].x * width)
        cy = int(face_landmarks.landmark[dic].y * height)
        pointarr.append([cx,cy])
    image_pts = np.array(pointarr,dtype=np.float32)
    #image_pts = np.float32(pointarr)
    # image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
    #                         shape[39], shape[42], shape[45], shape[31], shape[35],
    #                         shape[48], shape[54], shape[57], shape[8]])
    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))# 以8行2列显示

    # 计算欧拉角calc euler angle
    # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)#罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))# 水平拼接，vconcat垂直拼接
    # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    
    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
 
 
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    return reprojectdst, euler_angle# 投影误差，欧拉角

#计算左、右眼EAR值以及平均值
def computerEar(face_landmarks,height,width):
    dics = [33,161,157,133,154,163,362,384,388,263,390,381]
    pointarr = []
    for dic in dics:
        cx = int(face_landmarks.landmark[dic].x * width)
        cy = int(face_landmarks.landmark[dic].y * height)
        pointarr.append([cx,cy])
        #cv2.circle(img, (cx, cy), 5, (0, 0, 255))
        #img = cv2.putText(img,str(dic),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1*scaler,(0,255,0),1)
    pointarr = np.array(pointarr)

    #计算左眼ear
    A = np.linalg.norm(pointarr[1]-pointarr[5])
    B = np.linalg.norm(pointarr[2]-pointarr[4])
    C = np.linalg.norm(pointarr[0]-pointarr[3])
    leftear = (A + B) / (2.0 * C) 
    leftear = round(leftear,2)
    #计算右眼ear
    D = np.linalg.norm(pointarr[7]-pointarr[11])
    E = np.linalg.norm(pointarr[8]-pointarr[10])
    F = np.linalg.norm(pointarr[6]-pointarr[9])
    rightear = (D + E) / (2.0 * F) 
    rightear = round(rightear,2)
    avaear = (leftear+rightear)/2
    return leftear,rightear,avaear
def computermouthear(face_landmarks,height,width):
    #dics = [62,39,269,306,405,181]
    dics = [62,80,310,306,318,88]
    pointarr = []
    for dic in dics:
        cx = int(face_landmarks.landmark[dic].x * width)
        cy = int(face_landmarks.landmark[dic].y * height)
        pointarr.append([cx,cy])
        #cv2.circle(img, (cx, cy), 5, (0, 0, 255))
        #img = cv2.putText(img,str(dic),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1*scaler,(0,255,0),1)
    pointarr = np.array(pointarr)
    #计算Mouth ear
    A = np.linalg.norm(pointarr[1]-pointarr[5])
    B = np.linalg.norm(pointarr[2]-pointarr[4])
    C = np.linalg.norm(pointarr[0]-pointarr[3])
    mouthear = (A + B) / (2.0 * C) 
    mouthear = round(mouthear,2)
    return mouthear

# 定义常数
# 眼睛长宽比
# 闪烁阈值
EYE_AR_THRESH = 0.1
EYE_AR_CONSEC_FRAMES = 3
# 打哈欠长宽比
# 闪烁阈值
MAR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3
# 瞌睡点头
HAR_THRESH = 0.3
NOD_AR_CONSEC_FRAMES = 3
# 初始化帧计数器和眨眼总数
COUNTER = 0
TOTAL = 0
# 初始化帧计数器和打哈欠总数
mCOUNTER = 0
mTOTAL = 0
# 初始化帧计数器和点头总数
hCOUNTER = 0
hTOTAL = 0
# 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
print("[INFO] loading Mediapipe FaceMesh predictor...")

# scaler = 1
# # 第一步：打开cv2 本地摄像头
# cap = cv2.VideoCapture(0)
# cap.open(0)
# # 从视频流循环帧
# while True:
#     success ,frame = cap.read()
#     frame = imutils.resize(frame, width=720)
#     if not success:
#         print('ERROR')
#         break
#     h,w = frame.shape[0],frame.shape[1]
#     #BGR转RGB
#     img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

#     results = model.process(img)
#     frame = predict.predictyolo(frame)
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             #绘制人脸网格
#             mp_drawing.draw_landmarks(
#                 image = img,
#                 landmark_list = face_landmarks,
#                 #connections = mp_face_mesh.FACEMESH_TESSELATION,# 可视化Face Mesh
#                 connections = mp_face_mesh.FACEMESH_CONTOURS,# 可视化脸轮廓
#                 #landmark_drawing_spec 为关键点可视化样式 None为默认值（不显示关键点）
#                 #landmark_drawing_spec= mp_drawing.DrawingSpec(thickness=1,circle_radius=2,color=[66,77,229])
#                 landmark_drawing_spec = landmark_drawing_spec,#关键点圆圈样式
#                 connection_drawing_spec = connection_drawing_spec#轮廓样式
#             )
        
#             #img = cv2.putText(img,'Face Detected',(25*scaler,50*scaler),cv2.FONT_HERSHEY_SIMPLEX,1.25*scaler,(255,0,255),2*scaler)
#             leftEar,rightEar,averageEar =computerEar(face_landmarks,h,w)
#             #cv2.putText(img, "Left Eye Aspect Ratio:{}".format(str(leftEar)), (30*scaler,150*scaler), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
#             #cv2.putText(img, "Right Eye Aspect Ratio:{}".format(str(rightEar)), (30*scaler,200*scaler), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
#             #cv2.putText(img, "Average Eye Aspect Ratio:{}".format(str(averageEar)), (30*scaler,250*scaler), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
#             mouthear = computermouthear(face_landmarks,h,w)
#             #cv2.putText(img, "Mouth Aspect Ratio:{}".format(str(mouthear)), (30*scaler,300*scaler), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

#             '''
#             分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示进行了一次眨眼活动
#             '''
#             # 第十三步：循环，满足条件的，眨眼次数+1
#             if averageEar < EYE_AR_THRESH:# 眼睛长宽比：0.2
#                 COUNTER += 1
#             else:
#                 # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
#                 if COUNTER >= EYE_AR_CONSEC_FRAMES:# 阈值：3
#                     TOTAL += 1
#                 # 重置眼帧计数器
#                 COUNTER = 0
                
#             # 第十四步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
#             cv2.putText(frame, "Faces: {}".format(len(results.multi_face_landmarks)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)     
#             cv2.putText(frame, "COUNTER: {}".format(COUNTER), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
#             cv2.putText(frame, "EAR: {:.2f}".format(averageEar), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             cv2.putText(frame, "Blinks: {}".format(TOTAL), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
#             '''
#                 计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
#             '''
#             # 同理，判断是否打哈欠    
#             if mouthear > MAR_THRESH:# 张嘴阈值0.5
#                 mCOUNTER += 1
#                 cv2.putText(frame, "Yawning!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             else:
#                 # 如果连续3次都小于阈值，则表示打了一次哈欠
#                 if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:# 阈值：3
#                     mTOTAL += 1
#                 # 重置嘴帧计数器
#                 mCOUNTER = 0
#             cv2.putText(frame, "COUNTER: {}".format(mCOUNTER), (150, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
#             cv2.putText(frame, "MAR: {:.2f}".format(mouthear), (300, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             cv2.putText(frame, "Yawning: {}".format(mTOTAL), (450, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
#             """
#             瞌睡点头
#             """
#             # 第十五步：获取头部姿态
#             reprojectdst, euler_angle = get_head_pose(face_landmarks,w,h)#TODO
#             har = euler_angle[0, 0]# 取pitch旋转角度
#             if har > HAR_THRESH:# 点头阈值0.3
#                 hCOUNTER += 1
#             else:
#                 # 如果连续3次都小于阈值，则表示瞌睡点头一次
#                 if hCOUNTER >= NOD_AR_CONSEC_FRAMES:# 阈值：3
#                     hTOTAL += 1
#                 # 重置点头帧计数器
#                 hCOUNTER = 0
#             # 绘制正方体12轴
#             for start, end in line_pairs:
#                 cv2.line(frame, (int(reprojectdst[start][0]),int(reprojectdst[start][1])), (int(reprojectdst[end][0]),int(reprojectdst[end][1])),(0, 0, 255))
#                 #cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
#             # 显示角度结果
#             cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), thickness=2)# GREEN
#             cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (150, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 0, 0), thickness=2)# BLUE
#             cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (300, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), thickness=2)# RED    
#             cv2.putText(frame, "Nod: {}".format(hTOTAL), (450, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
#     else:
#         frame = cv2.putText(frame,"No Face Detected",(25*scaler,50*scaler),cv2.FONT_HERSHEY_SIMPLEX,1.25*scaler,(255,0,255),2*scaler)
#     #print('嘴巴实时长宽比:{:.2f} '.format(mouthear)+"\t是否张嘴："+str([False,True][mouthear > MAR_THRESH]))
#     #print('眼睛实时长宽比:{:.2f} '.format(averageEar)+"\t是否眨眼："+str([False,True][COUNTER>=1]))

#     # 确定疲劳提示:眨眼50次，打哈欠15次，瞌睡点头15次
#     if TOTAL >= 50 or mTOTAL>=15 or hTOTAL>=15:
#         cv2.putText(frame, "SLEEP!!!", (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    
#     # 按q退出
#     cv2.putText(frame, "Press 'q': Quit", (20, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
#     # 窗口显示 show with opencv
#     cv2.imshow("Frame", frame)
    
#     # if the `q` key was pressed, break from the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放摄像头 release camera
# cap.release()
# # do a bit of cleanup
# cv2.destroyAllWindows()

def detectTiredMethod(frame):
    origin_width = frame.shape[1]
    frame = imutils.resize(frame, width=720)
    h,w = frame.shape[0],frame.shape[1]
    #BGR转RGB
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = model.process(img)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #绘制人脸网格
            mp_drawing.draw_landmarks(
                image = img,
                landmark_list = face_landmarks,
                #connections = mp_face_mesh.FACEMESH_TESSELATION,# 可视化Face Mesh
                connections = mp_face_mesh.FACEMESH_CONTOURS,# 可视化脸轮廓
                #landmark_drawing_spec 为关键点可视化样式 None为默认值（不显示关键点）
                #landmark_drawing_spec= mp_drawing.DrawingSpec(thickness=1,circle_radius=2,color=[66,77,229])
                landmark_drawing_spec = landmark_drawing_spec,#关键点圆圈样式
                connection_drawing_spec = connection_drawing_spec#轮廓样式
            )
        
            #img = cv2.putText(img,'Face Detected',(25*scaler,50*scaler),cv2.FONT_HERSHEY_SIMPLEX,1.25*scaler,(255,0,255),2*scaler)
            leftEar,rightEar,averageEar =computerEar(face_landmarks,h,w)
            #cv2.putText(img, "Left Eye Aspect Ratio:{}".format(str(leftEar)), (30*scaler,150*scaler), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            #cv2.putText(img, "Right Eye Aspect Ratio:{}".format(str(rightEar)), (30*scaler,200*scaler), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            #cv2.putText(img, "Average Eye Aspect Ratio:{}".format(str(averageEar)), (30*scaler,250*scaler), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            mouthear = computermouthear(face_landmarks,h,w)
            #cv2.putText(img, "Mouth Aspect Ratio:{}".format(str(mouthear)), (30*scaler,300*scaler), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            '''
            分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示进行了一次眨眼活动
            '''
            # 第十三步：循环，满足条件的，眨眼次数+1
            global COUNTER,TOTAL,mCOUNTER,mTOTAL,hCOUNTER,hTOTAL
            if averageEar < EYE_AR_THRESH:# 眼睛长宽比：0.2
                COUNTER += 1
            else:
                # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                if COUNTER >= EYE_AR_CONSEC_FRAMES:# 阈值：3
                    TOTAL += 1
                # 重置眼帧计数器
                COUNTER = 0
                
            # 第十四步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
            cv2.putText(frame, "Faces: {}".format(len(results.multi_face_landmarks)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)     
            cv2.putText(frame, "COUNTER: {}".format(COUNTER), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
            cv2.putText(frame, "EAR: {:.2f}".format(averageEar), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            '''
                计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
            '''
            # 同理，判断是否打哈欠    
            if mouthear > MAR_THRESH:# 张嘴阈值0.5
                mCOUNTER += 1
                cv2.putText(frame, "Yawning!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # 如果连续3次都小于阈值，则表示打了一次哈欠
                if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:# 阈值：3
                    mTOTAL += 1
                # 重置嘴帧计数器
                mCOUNTER = 0
            cv2.putText(frame, "COUNTER: {}".format(mCOUNTER), (150, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
            cv2.putText(frame, "MAR: {:.2f}".format(mouthear), (300, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Yawning: {}".format(mTOTAL), (450, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            """
            瞌睡点头
            """
            # 第十五步：获取头部姿态
            reprojectdst, euler_angle = get_head_pose(face_landmarks,w,h)#TODO
            
            har = euler_angle[0, 0]# 取pitch旋转角度
            if har > HAR_THRESH:# 点头阈值0.3
                hCOUNTER += 1
            else:
                # 如果连续3次都小于阈值，则表示瞌睡点头一次
                if hCOUNTER >= NOD_AR_CONSEC_FRAMES:# 阈值：3
                    hTOTAL += 1
                # 重置点头帧计数器
                hCOUNTER = 0
            # 绘制正方体12轴
            for start, end in line_pairs:
                #cv2.line(frame, (reprojectdst[start][0].astype('int64'),reprojectdst[start][1].astype('int64')), (reprojectdst[end][0].astype('int64'),reprojectdst[end][1].astype('int64')),(0, 0, 255))
                #print("start:",reprojectdst[start])
                #print("end:",reprojectdst[end])
                cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
            # 显示角度结果
            cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), thickness=2)# GREEN
            cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (150, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 0, 0), thickness=2)# BLUE
            cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (300, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), thickness=2)# RED    
            cv2.putText(frame, "Nod: {}".format(hTOTAL), (450, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    else:
        frame = cv2.putText(frame,"No Face Detected",(25,50),cv2.FONT_HERSHEY_SIMPLEX,1.25,(255,0,255),2)
    #print('嘴巴实时长宽比:{:.2f} '.format(mouthear)+"\t是否张嘴："+str([False,True][mouthear > MAR_THRESH]))
    #print('眼睛实时长宽比:{:.2f} '.format(averageEar)+"\t是否眨眼："+str([False,True][COUNTER>=1]))

    # 确定疲劳提示:眨眼50次，打哈欠15次，瞌睡点头15次
    if TOTAL >= 50 or mTOTAL>=15 or hTOTAL>=15:
        cv2.putText(frame, "SLEEP!!!", (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
    # 按q退出
    cv2.putText(frame, "Press 'q': Quit", (20, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
    # 窗口显示 show with opencv
    #cv2.imshow("Frame", frame)
    frame = imutils.resize(frame, width=origin_width)
    return frame

