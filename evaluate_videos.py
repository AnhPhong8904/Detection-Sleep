import os
import cv2
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.video_processor import VideoProcessor
from utils.config import MIN_FACE_DETECTION_CONFIDENCE, NUM_FACES, WINDOW_NAME, MIN_FACE_PRESENCE_CONFIDENCE, MIN_TRACKING_CONFIDENCE


def build_detector(model_path: str = "face_landmarker.task"):
	"""Create a MediaPipe Face Landmarker detector like in main.py"""
	base_options = python.BaseOptions(model_asset_path=model_path)
	options = vision.FaceLandmarkerOptions(
		base_options=base_options,
		min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
		min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
		min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
		output_face_blendshapes=True,
		output_facial_transformation_matrixes=True,
		num_faces=NUM_FACES,
		running_mode=vision.RunningMode.VIDEO,
	)
	return vision.FaceLandmarker.create_from_options(options)


def predict_video_label(video_path: str, show: bool = False, save_dir: str = None, min_events: int = 1) -> str:
	"""Predict a single video's label ('ngu_gat' or 'tinh_tao') using existing VideoProcessor logic.
	Video-level rule: if DROWSY_HEAD events >= min_events → ngu_gat
	"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"❌ Không thể mở video: {video_path}")
		return "unknown"

	# Create a fresh detector per video to avoid MediaPipe VIDEO mode timestamp constraints
	detector = build_detector()

	# Optional writer for visualization
	video_writer = None
	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		fps = cap.get(cv2.CAP_PROP_FPS) or 30
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		out_path = os.path.join(save_dir, os.path.basename(video_path))
		video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

	processor = VideoProcessor(detector, video_writer=video_writer)
	processor.start_processing()

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		timestamp = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
		output = processor.process_frame(frame, timestamp)
		if show:
			cv2.imshow(WINDOW_NAME, output)
			key = cv2.waitKey(1) & 0xFF
			if key == 27:  # ESC to skip current video
				print("⏭️ Bỏ qua video theo yêu cầu (ESC)")
				break

	cap.release()
	processor.stop_processing()
	if show:
		cv2.destroyAllWindows()

	# Count DROWSY_HEAD events
	drowsy_events = [e for e in processor.tracker.detection_events if e['type'] == 'DROWSY_HEAD']
	return 'ngu_gat' if len(drowsy_events) >= min_events else 'tinh_tao'


def compute_metrics(gt: np.ndarray, pred: np.ndarray):
	"""Compute accuracy, precision, recall, f1 and confusion matrix."""
	tp = int(((gt == 1) & (pred == 1)).sum())
	fn = int(((gt == 1) & (pred == 0)).sum())
	fp = int(((gt == 0) & (pred == 1)).sum())
	tn = int(((gt == 0) & (pred == 0)).sum())
	accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
	precision = tp / max(1, (tp + fp))
	recall = tp / max(1, (tp + fn))
	f1 = (2 * precision * recall / max(1e-12, (precision + recall))) if (precision + recall) > 0 else 0.0
	cm = np.array([[tn, fp], [fn, tp]], dtype=int)
	return accuracy, precision, recall, f1, cm


def save_confusion_matrix(cm: np.ndarray, out_path: str):
	fig, ax = plt.subplots(figsize=(4, 4))
	im = ax.imshow(cm, cmap='Blues')
	for (i, j), v in np.ndenumerate(cm):
		ax.text(j, i, str(v), ha='center', va='center', color='black')
	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])
	ax.set_xticklabels(['Pred 0', 'Pred 1'])
	ax.set_yticklabels(['GT 0', 'GT 1'])
	ax.set_title('Confusion Matrix')
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(description='Evaluate videos with rule-based detection')
	parser.add_argument('--csv', type=str, default=r'csv\ground_truth1.csv', help='Path to ground truth CSV')
	parser.add_argument('--videos', type=str, default=r'sugawara_real_sample', help='Base videos directory')
	parser.add_argument('--outdir', type=str, default='csv', help='Output directory for CSV and reports')
	parser.add_argument('--show', action='store_true', help='Show inference window while processing')
	parser.add_argument('--save-vis', action='store_true', help='Save annotated videos to outdir/vis')
	parser.add_argument('--min-events', type=int, default=1, help='Min DROWSY_HEAD events to label as ngu_gat')
	args = parser.parse_args()

	# Read ground truth
	csv_path = args.csv
	if not os.path.exists(csv_path):
		print(f"❌ Không tìm thấy {csv_path}")
		return

	df = pd.read_csv(csv_path)
	# Support either: (video_name,label) with text labels or (video_name,ground_truth) with 1/0
	if 'video_name' not in df.columns:
		print("❌ ground_truth.csv cần có cột 'video_name'")
		return
	if 'label' in df.columns:
		gt_col = 'label'
	elif 'ground_truth' in df.columns:
		gt_col = 'ground_truth'
	else:
		print("❌ ground_truth.csv cần có cột 'label' hoặc 'ground_truth'")
		return

	predictions = []
	predictions_numeric = []
	for _, row in df.iterrows():
		video_name = row['video_name']
		# Try to find the video in known folders
		video_path = None
		for folder in ['drowsiness', 'normal']:
			candidate = os.path.join(args.videos, folder, video_name)
			if os.path.exists(candidate):
				video_path = candidate
				break
		if video_path is None:
			print(f"⚠️ Không tìm thấy file video cho: {video_name}")
			pred = 'unknown'
		else:
			print(f"🔎 Đang dự đoán: {video_name}")
			vis_dir = os.path.join(args.outdir, 'vis') if args.save_vis else None
			pred = predict_video_label(video_path, show=args.show, save_dir=vis_dir, min_events=args.min_events)
		predictions.append(pred)
		predictions_numeric.append(1 if pred == 'ngu_gat' else (0 if pred == 'tinh_tao' else -1))

	# Write augmented CSV
	os.makedirs(args.outdir, exist_ok=True)
	df_out = df.copy()
	df_out['predicted_label'] = predictions
	df_out['predicted'] = predictions_numeric
	out_csv = os.path.join(args.outdir, 'ground_truth_with_pred.csv')
	df_out.to_csv(out_csv, index=False)
	print(f"✅ Đã ghi {out_csv}")

	# Compute metrics (ignore unknown)
	# Build comparable ground truth numeric series
	if gt_col == 'label':
		gt_numeric = df_out['label'].map({'ngu_gat': 1, 'tinh_tao': 0})
	else:  # 'ground_truth' with 1/0
		gt_numeric = df_out['ground_truth']
	# Known predictions are 0/1
	mask_known = df_out['predicted'].isin([0, 1])
	df_known = df_out[mask_known].copy()
	gt_known = gt_numeric[mask_known]
	if not df_known.empty:
		pred_known = df_known['predicted'].values.astype(int)
		gt_known_vals = gt_known.values.astype(int)
		acc, prec, rec, f1, cm = compute_metrics(gt_known_vals, pred_known)
		# Write report
		report_path = os.path.join(args.outdir, 'evaluation_accuracy.txt')
		with open(report_path, 'w', encoding='utf-8') as f:
			f.write(f"Total videos (known): {len(df_known)}\n")
			f.write(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)\n")
			f.write(f"Precision: {prec:.4f} ({prec*100:.2f}%)\n")
			f.write(f"Recall:    {rec:.4f} ({rec*100:.2f}%)\n")
			f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n")
			f.write("\nConfusion Matrix (rows=GT, cols=Pred)\n")
			f.write(f"[[{cm[0,0]} {cm[0,1]}]\n [{cm[1,0]} {cm[1,1]}]]\n")
		print(f"📄 Đã ghi {report_path}")
		print(f"🎯 Accuracy:  {acc:.2%} | Precision: {prec:.2%} | Recall: {rec:.2%} | F1: {f1:.2%}")
		# Save confusion matrix plot
		cm_path = os.path.join(args.outdir, 'confusion_matrix.png')
		save_confusion_matrix(cm, cm_path)
		print(f"🖼️  Đã lưu biểu đồ: {cm_path}")
	else:
		print("⚠️ Không có dự đoán hợp lệ để tính metrics")


if __name__ == '__main__':
	main()
