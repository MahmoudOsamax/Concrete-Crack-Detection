from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import math
import logging

logger = logging.getLogger(__name__)

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def extract_bboxes(fused):
    """Compute bounding boxes from masks."""
    try:
        mask = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
        mask[mask < 40] = 0
        mask[mask >= 40] = 1
        mask = mask.reshape(fused.shape[0], fused.shape[1], 1)
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        
        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        vertical_indicies = np.where(np.any(mask, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[0] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)
    except Exception as e:
        logger.error(f"Error in extract_bboxes: {str(e)}")
        return np.zeros([1, 4], dtype=np.int32)
def getContours(npImage, overlay_img, realHeight, realWidth, unit, confidence, angle_th=30):
    """Process image to detect contours and annotate measurements."""
    try:
        image = npImage.copy()
        imgHeight, imgWidth = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(gray, 50, 80)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if not cnts:
            logger.warning("No contours found")
            return overlay_img

        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetricHeight = realHeight / imgHeight
        pixelsPerMetricWidth = realWidth / imgWidth

        y1, x1, y2, x2 = extract_bboxes(npImage)[0]
        cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for c in cnts:
            if cv2.contourArea(c) < 100:
                continue

            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.line(overlay_img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 0, 255), 1)
            top_p = min([(int(tlblX), int(tlblY)), (int(trbrX), int(trbrY))], key=lambda x: x[1])
            bot_p = max([(int(tlblX), int(tlblY)), (int(trbrX), int(trbrY))], key=lambda x: x[1])
            D_ad = ((top_p[1] - bot_p[1]) ** 2 + (top_p[0] - bot_p[0]) ** 2) ** 0.5 + 1e-7

            P1 = min(top_p, bot_p, key=lambda x: x[0])
            P2 = max(top_p, bot_p, key=lambda x: x[0])
            slope = (P1[1] - P2[1]) / (P2[0] - P1[0]) if (P2[0] - P1[0]) != 0 else 0
            cat = ''
            angle = 0
            if slope > 0:
                angle = np.arccos((top_p[0] - bot_p[0]) / D_ad) * 180 / math.pi
                cv2.putText(overlay_img, f"angle={angle:.1f}", (max(top_p[0] - 100, 0), top_p[1] + 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 1)
            else:
                angle = np.arccos((bot_p[0] - top_p[0]) / D_ad) * 180 / math.pi
                cv2.putText(overlay_img, f"angle={angle:.1f}", (top_p[0], top_p[1] + 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 1)
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            length = cv2.arcLength(c, True) / 2. * pixelsPerMetricWidth
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            mask = gray.copy()
            mask[mask < 40] = 0
            width = cv2.countNonZero(mask[cY][:])
            if width > 0:
                right_most_x = np.max(np.nonzero(mask[cY][:]))
                left_most_x = np.min(np.nonzero(mask[cY][:]))
                cv2.line(overlay_img, (int(left_most_x), int(cY)), (int(right_most_x), int(cY)), (0, 0, 255), 1)
                width *= pixelsPerMetricWidth
            else:
                width = 0

            if angle < angle_th:
                cat += 'H'
                cv2.putText(overlay_img, f"L={length:.1f}{unit}", (int(tltrX), int(tltrY) + 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 1)
                cv2.putText(overlay_img, f"W={width:.1f}{unit}", (int(tltrX), int(tltrY) + 55),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 1)
            else:
                cat += 'V'
                cv2.putText(overlay_img, f"L={length:.1f}{unit}", (int(tltrX), int(tltrY)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 1)
                cv2.putText(overlay_img, f"W={width:.1f}{unit}", (int(tltrX), int(tltrY) + 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 1)
            if slope > 0:
                cat += 'L'
            else:
                cat += 'R'
            cv2.putText(overlay_img, f"Crack {confidence.item()*100:.2f}% cat={cat}", (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (36, 255, 12), 1)
        return overlay_img
    except Exception as e:
        logger.error(f"Error in getContours: {str(e)}")
        return overlay_img