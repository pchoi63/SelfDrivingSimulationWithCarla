import cv2
import numpy as np
import math



def sumMatrix(A, B):
    A = np.array(A)
    B = np.array(B)
    answer = A + B
    return answer.tolist()


def laneDetect(roi_im, size_im, tspeed, cspeed, lstate):
            
    pt1_sum_ri = (0, 0)
    pt2_sum_ri = (0, 0)
    pt1_avg_ri = (0, 0)
    count_posi_num_ri = 0
    pt1_sum_le = (0, 0)
    pt2_sum_le = (0, 0)
    pt1_avg_le = (0, 0)
    count_posi_num_le = 0
    org = (320, 100)
    torg = (10, 20)
    corg = (10, 40)
    lorg = (10, 60)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #cv2.putText(size_im, hehe, org, font, 0.7, (0, 0, 255), 2)

    #cv2.imshow("plz", trafficImage)
    #################################################
    # Gaussian Blur Filter
    Blur_im = cv2.bilateralFilter(roi_im, d=-1, sigmaColor=3, sigmaSpace=3)
    #################################################

    #################################################
    # Canny edge detector
    edges = cv2.Canny(Blur_im, 50, 100)
    #cv2.imshow("edges", edges)
    #################################################

    #################################################
    # Hough Transformation
    #lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=80, minLineLength=30, maxLineGap=50)
    # rho, theta는 1씩 변경하면서 검출하겠다는 의미, np.pi/180 라디안 = 1'
    # threshold 숫자가 작으면 정밀도↓ 직선검출↑, 크면 정밀도↑ 직선검출↓
    # min_line_len 선분의 최소길이
    # max_line,gap 선분 사이의 최대 거리
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=22, minLineLength=10, maxLineGap=20)

    N = lines.shape[0]

    for line in range(N):
        # for line in lines:

        # x1, y1, x2, y2 = line[0]

        x1 = lines[line][0][0]
        y1 = lines[line][0][1]
        x2 = lines[line][0][2]
        y2 = lines[line][0][3]

        if x2 == x1:
            a = 1
        else:
            a = x2 - x1

        b = y2 - y1

        radi = b / a 

        theta_atan = math.atan(radi) * 180.0 / math.pi
        # print('theta_atan=', theta_atan)

        pt1_ri = (x1 + 108, y1 + 240)
        pt2_ri = (x2 + 108, y2 + 240)
        pt1_le = (x1 + 108, y1 + 240)
        pt2_le = (x2 + 108, y2 + 240)

        if theta_atan > 30.0 and theta_atan < 80.0:
            count_posi_num_ri += 1

            pt1_sum_ri = sumMatrix(pt1_ri, pt1_sum_ri)
            pt2_sum_ri = sumMatrix(pt2_ri, pt2_sum_ri)
        if theta_atan < -30.0 and theta_atan > -80.0:
            count_posi_num_le += 1

            pt1_sum_le = sumMatrix(pt1_le, pt1_sum_le)
            pt2_sum_le = sumMatrix(pt2_le, pt2_sum_le)
    
    pt1_avg_ri = pt1_sum_ri // np.array(count_posi_num_ri)
    pt2_avg_ri = pt2_sum_ri // np.array(count_posi_num_ri)
    pt1_avg_le = pt1_sum_le // np.array(count_posi_num_le)
    pt2_avg_le = pt2_sum_le // np.array(count_posi_num_le)

    #################################################
    # 차석인식의 흔들림 보정
    # right-----------------------------------------------------------
    x1_avg_ri, y1_avg_ri = pt1_avg_ri
    x2_avg_ri, y2_avg_ri = pt2_avg_ri

    a_avg_ri = ((y2_avg_ri - y1_avg_ri) / (x2_avg_ri - x1_avg_ri))
    b_avg_ri = (y2_avg_ri - (a_avg_ri * x2_avg_ri))

    pt2_y2_fi_ri = 480

    if a_avg_ri > 0:
        pt2_x2_fi_ri = int((pt2_y2_fi_ri - b_avg_ri) // a_avg_ri)
    else:
        pt2_x2_fi_ri = 0

    pt2_fi_ri = (pt2_x2_fi_ri, pt2_y2_fi_ri)

    # left------------------------------------------------------------
    x1_avg_le, y1_avg_le = pt1_avg_le
    x2_avg_le, y2_avg_le = pt2_avg_le

    a_avg_le = ((y2_avg_le - y1_avg_le) / (x2_avg_le - x1_avg_le))
    b_avg_le = (y2_avg_le - (a_avg_le * x2_avg_le))

    pt1_y1_fi_le = 480
    if a_avg_le < 0:
        pt1_x1_fi_le = int((pt1_y1_fi_le - b_avg_le) // a_avg_le)
    else:
        pt1_x1_fi_le = 0

    pt1_fi_le = (pt1_x1_fi_le, pt1_y1_fi_le)
    #################################################

    #################################################
    # lane painting
    
    #################################################

    #################################################
    # possible lane

    #################################################
    # lane center 및 steering 계산 (320, 360)
    lane_center_y_ri = 360
    if a_avg_ri > 0:
        lane_center_x_ri = int((lane_center_y_ri - b_avg_ri) // a_avg_ri)
    else:
        lane_center_x_ri = 0

    lane_center_y_le = 360
    if a_avg_le < 0:
        lane_center_x_le = int((lane_center_y_le - b_avg_le) // a_avg_le)
    else:
        lane_center_x_le = 0

    laneDiff = lane_center_x_ri - lane_center_x_le

    if laneDiff < 0 or laneDiff <= 120 or laneDiff >= 360:
        lane = False
    else:
        lane = True

    org = (320, 440)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(size_im, (320, 350), (320, 370), (0, 228, 255), 1)
    
    if lane == True:
        # right-----------------------------------------------------------
        cv2.line(size_im, tuple(pt1_avg_ri), tuple(pt2_fi_ri), (0, 255, 0), 2)  # right lane
        # left-----------------------------------------------------------
        cv2.line(size_im, tuple(pt1_fi_le), tuple(pt2_avg_le), (0, 255, 0), 2)  # left lane


        # caenter left lane (255, 90, 185)
        cv2.line(size_im, (lane_center_x_le, lane_center_y_le - 10), (lane_center_x_le, lane_center_y_le + 10), (0, 228, 255), 1)
        #print("x"+str(lane_center_x_le))#180
        #print("y"+str(lane_center_y_le))#360
        # caenter right lane
        cv2.line(size_im, (lane_center_x_ri, lane_center_y_ri - 10), (lane_center_x_ri, lane_center_y_ri + 10), (0, 228, 255), 1)
        #print("x"+str(lane_center_x_ri))#480
        #print("y"+str(lane_center_y_ri))#360

        FCP_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8) + 0
        FCP = np.array([pt2_avg_le, pt1_fi_le, pt2_fi_ri, pt1_avg_ri])
        cv2.fillConvexPoly(FCP_img, FCP, color=(255, 242, 213))  # BGR
        alpha = 0.9
        size_im = cv2.addWeighted(size_im, alpha, FCP_img, 1 - alpha, 0)

        # caenter middle lane
        lane_center_x = ((lane_center_x_ri - lane_center_x_le) // 2) + lane_center_x_le
        cv2.line(size_im, (lane_center_x, lane_center_y_ri - 10), (lane_center_x, lane_center_y_le + 10), (0, 228, 255), 1)

        # center-----------------------------------------------------------
        cv2.line(size_im, (320, 480),(lane_center_x, lane_center_y_le + 10) , (0, 228, 255), 1)  # middle lane
        diff = 320 - lane_center_x
        cv2.putText(size_im, str(diff), org, font, 0.7, (0, 255, 100), 2)
        steer = diff / - 1100
    
    else:
        # right-----------------------------------------------------------
        cv2.line(size_im, tuple((640, 480)), tuple((480, 360)), (0, 255, 0), 2)  # right lane
        # left-----------------------------------------------------------
        cv2.line(size_im, tuple((0, 480)), tuple((180, 360)), (0, 255, 0), 2)  # left lane
        

        cv2.putText(size_im, "0", org, font, 0.7, (0, 255, 100), 2)
        steer = 0
        cv2.line(size_im, (180, 350), (180, 370), (0, 228, 255), 1)
        
        # caenter right lane
        cv2.line(size_im, (480,350), (480, 370), (0, 228, 255), 1)
        #print("x"+str(lane_center_x_ri))#480
        #print("y"+str(lane_center_y_ri))#360

        FCP_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8) + 0
        FCP = np.array([(180, 360),(0, 480),(640, 480),(480, 360)])
        cv2.fillConvexPoly(FCP_img, FCP, color=(255, 242, 213))  # BGR
        alpha = 0.9
        size_im = cv2.addWeighted(size_im, alpha, FCP_img, 1 - alpha, 0)

        # caenter middle lane
        lane_center_x = 320
        cv2.line(size_im, (320, 350), (320, 370), (0, 228, 255), 1)

        # center-----------------------------------------------------------
        cv2.line(size_im, (320, 480),(320, 370) , (0, 228, 255), 1)
    cv2.rectangle(size_im, (0,0), (250, 75), (167,225,210), -1)
    #print(cspeed)
    cv2.putText(size_im, "Current Speed: " + str(round(cspeed, 2)), corg, font, 0.7, (0,0,0), 2)
    cv2.putText(size_im, "Target Speed: " + str(tspeed), torg, font, 0.7, (0,0,0), 2)
    cv2.putText(size_im, "Light State: " + lstate, lorg, font, 0.7, (0,0,0), 2)
    
    #################################################

    # 변수 초기화
    count_posi_num_ri = 0

    pt1_sum_ri = (0, 0)
    pt2_sum_ri = (0, 0)
    pt1_avg_ri = (0, 0)
    pt2_avg_ri = (0, 0)

    count_posi_num_le = 0

    pt1_sum_le = (0, 0)
    pt2_sum_le = (0, 0)
    pt1_avg_le = (0, 0)
    pt2_avg_le = (0, 0)

    cv2.imshow('frame_size_im', size_im)
    cv2.waitKey(2)

    return steer