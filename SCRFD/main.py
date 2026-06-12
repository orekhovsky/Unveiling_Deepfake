import argparse
from pathlib import Path

import cv2

from nets.nn import FaceDetector

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def draw_boxes(frame, boxes):
    for box in boxes.astype('int32'):
        x_min, y_min, x_max, y_max, _ = box
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 255), 1)

        cv2.line(frame, (int(x_min), int(y_min)), (int(x_min + 15), int(y_min)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_min), int(y_min)), (int(x_min), int(y_min + 15)), (255, 0, 255), 3)

        cv2.line(frame, (int(x_max), int(y_max)), (int(x_max - 15), int(y_max)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_max), int(y_max)), (int(x_max), int(y_max - 15)), (255, 0, 255), 3)

        cv2.line(frame, (int(x_max - 15), int(y_min)), (int(x_max), int(y_min)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_max), int(y_min)), (int(x_max), int(y_min + 15)), (255, 0, 255), 3)

        cv2.line(frame, (int(x_min), int(y_max - 15)), (int(x_min), int(y_max)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_min), int(y_max)), (int(x_min + 15), int(y_max)), (255, 0, 255), 3)
    return frame


def default_output_path(input_path: Path, is_image: bool) -> Path:
    suffix = '.jpg' if is_image else '.mp4'
    return input_path.with_name(f'{input_path.stem}_faces{suffix}')


def resolve_input(raw_input: str):
    if raw_input.lstrip('-').isdigit():
        return int(raw_input), 'camera'
    path = Path(raw_input)
    if not path.exists():
        raise FileNotFoundError(f'Input not found: {path}')
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return path, 'image'
    return path, 'video'


def process_frame(detector, frame, input_size):
    boxes, _ = detector.detect(frame, input_size=input_size)
    return draw_boxes(frame, boxes)


def run_image(detector, input_path: Path, output_path: Path, input_size):
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise RuntimeError(f'Failed to read image: {input_path}')

    frame = process_frame(detector, frame, input_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f'Failed to write image: {output_path}')
    print(f'Saved: {output_path}')


def run_video(detector, input_source, output_path: Path, input_size):
    stream = cv2.VideoCapture(input_source)
    if not stream.isOpened():
        raise RuntimeError(f'Failed to open video source: {input_source}')

    fps = stream.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        ok, first_frame = stream.read()
        if not ok:
            stream.release()
            raise RuntimeError('Failed to read first frame')
        height, width = first_frame.shape[:2]
        stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        first_frame = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        stream.release()
        raise RuntimeError(f'Failed to open VideoWriter: {output_path}')

    frame_idx = 0
    if first_frame is not None:
        writer.write(process_frame(detector, first_frame, input_size))
        frame_idx += 1

    while True:
        ok, frame = stream.read()
        if not ok:
            break
        writer.write(process_frame(detector, frame, input_size))
        frame_idx += 1

    stream.release()
    writer.release()
    print(f'Saved: {output_path} ({frame_idx} frames)')


def main():
    parser = argparse.ArgumentParser(description='SCRFD face detection (headless output)')
    parser.add_argument(
        '--input',
        '--cam_id',
        dest='input',
        required=True,
        help='Image path, video path, or camera index (e.g. 0)',
    )
    parser.add_argument('--output', '-o', default=None, help='Output .jpg or .mp4 path')
    parser.add_argument('--model', default='weights/model_1.onnx', help='ONNX model path')
    parser.add_argument('--input_size', type=int, nargs=2, default=[640, 640], metavar=('W', 'H'))
    args = parser.parse_args()

    input_source, input_kind = resolve_input(str(args.input))
    detector = FaceDetector(onnx_file=args.model)
    input_size = tuple(args.input_size)

    if input_kind == 'image':
        input_path = Path(input_source)
        output_path = Path(args.output) if args.output else default_output_path(input_path, is_image=True)
        run_image(detector, input_path, output_path, input_size)
        return

    if input_kind == 'video':
        input_path = Path(input_source)
        output_path = Path(args.output) if args.output else default_output_path(input_path, is_image=False)
        run_video(detector, str(input_path), output_path, input_size)
        return

    if args.output is None:
        raise ValueError('Camera input requires --output path (e.g. -o output.mp4)')
    run_video(detector, input_source, Path(args.output), input_size)


if __name__ == '__main__':
    main()
