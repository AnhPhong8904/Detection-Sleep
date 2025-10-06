import cv2


def draw_alert(frame, text="ALERT", color=(0, 0, 255)):
    """Draw a simple top banner alert with given text on a BGR frame.

    Args:
        frame: OpenCV BGR image (numpy array)
        text: Alert text to display
        color: BGR color tuple for the banner and text accent
    Returns:
        The frame with the alert overlay (in-place modification also occurs).
    """
    if frame is None:
        return frame

    height, width = frame.shape[:2]
    banner_height = max(30, int(0.08 * height))

    # Draw semi-transparent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), color, thickness=-1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Put text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = 10
    text_y = int(banner_height * 0.7)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame


def draw_bbox_with_label(frame, bbox, color=(0, 255, 0), label=None):
    """Draw a rectangle and optional label.

    Args:
        frame: OpenCV BGR image (numpy array)
        bbox: (x1, y1, x2, y2) in pixel coordinates
        color: BGR color tuple
        label: Optional string to show above the box
    Returns:
        frame with drawing applied.
    """
    if frame is None or bbox is None:
        return frame

    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = max(0, int(x2))
    y2 = max(0, int(y2))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size
        pad = 4
        # Label background
        cv2.rectangle(
            frame,
            (x1, max(0, y1 - text_h - baseline - 2 * pad)),
            (x1 + text_w + 2 * pad, y1),
            color,
            thickness=-1,
        )
        # Text in white
        cv2.putText(
            frame,
            label,
            (x1 + pad, y1 - baseline - pad),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return frame

