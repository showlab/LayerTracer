import cv2
import numpy as np
import os
import vtracer
import argparse

def parse_arguments():
    """
    Parse command-line arguments for customizing conversion parameters.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Layer Tracer Image Processing Script')

    #Input and Output
    parser.add_argument('--input', type=str, default='emoji_1.png',
                        help='Input image path (default: emoji_1.png)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: based on input image name)')

    # SVG Conversion Parameters
    parser.add_argument('--colormode', type=str, default='color',
                        choices=['color', 'binary'],
                        help='Color mode for SVG conversion')
    parser.add_argument('--mode', type=str, default='spline',
                        choices=['spline', 'polygon', 'none'],
                        help='Tracing mode for SVG conversion')
    parser.add_argument('--filter_speckle', type=int, default=8,
                        help='Speckle filter threshold')
    parser.add_argument('--color_precision', type=int, default=6,
                        help='Color precision for SVG conversion')
    parser.add_argument('--corner_threshold', type=int, default=60,
                        help='Corner detection threshold')
    parser.add_argument('--length_threshold', type=float, default=4.0,
                        help='Length threshold for path simplification')
    parser.add_argument('--splice_threshold', type=int, default=45,
                        help='Splice threshold for path merging')
    parser.add_argument('--path_precision', type=int, default=3,
                        help='Path precision for SVG conversion')

    # Threshold Parameters
    parser.add_argument('--thresh', type=int, default=5,
                        help='Threshold value for image difference')

    return parser.parse_args()
def create_output_directory(image_path, custom_output_dir=None):
    """
    Create a unique output directory based on the input image name.

    Args:
        image_path (str): Path to the input image
        custom_output_dir (str, optional): Custom output directory path

    Returns:
        str: Path to the created output directory
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if custom_output_dir:
        output_dir = os.path.join(custom_output_dir, base_name)
    else:
        output_dir = os.path.join(os.getcwd(), base_name)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def split_image_to_4(image_path, output_dir=None):
    """
    Split the input image into 4 equal cells.

    Args:
        image_path (str): Path to the input image
        output_dir (str, optional): Custom output directory

    Returns:
        tuple: List of cell images and output directory path
    """

    output_dir = create_output_directory(image_path, output_dir)

    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    cell_height = height // 2
    cell_width = width // 2
    cells = []
    positions = [(0, 0), (0, 1), (1, 1), (1, 0)]

    for idx, (i, j) in enumerate(positions):
        cell = img[i * cell_height:(i + 1) * cell_height,
               j * cell_width:(j + 1) * cell_width]
        cells.append(cell)
        output_path = os.path.join(output_dir, f'cell_{idx + 1}.png')
        cv2.imwrite(output_path, cell)

    return cells, output_dir


def process_consecutive_images(img1_path, img2_path, output_number, output_dir, thresh=5):
    try:
        # Read images in grayscale and color
        img1_gray = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2_gray = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        img2_color = cv2.imread(img2_path)

        if img1_gray is None or img2_gray is None or img2_color is None:
            print(f"Unable to read images: {img1_path} or {img2_path}")
            return

        diff = cv2.absdiff(img1_gray, img2_gray)
        _, diff_thresh = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        clean_diff = clean_difference_image(diff_thresh)

        # （BGR + Alpha）
        result_rgba = cv2.cvtColor(img2_color, cv2.COLOR_BGR2BGRA)
        result_rgba[:, :, 3] = clean_diff

        result_path = os.path.join(output_dir, f'difference_{output_number}.png')
        cv2.imwrite(result_path, result_rgba)

        svg_path = os.path.join(output_dir, f'difference_{output_number}.svg')
        return result_path, svg_path

    except Exception as e:
        print(f"Error processing image pair {output_number}: {e}")
        return None, None

def clean_difference_image(diff_thresh):
    """
    Clean and filter the difference image to remove noise.

    Args:
        diff_thresh (numpy.ndarray): Thresholded difference image

    Returns:
        numpy.ndarray: Cleaned difference image
    """
    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    diff_opened = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
    diff_closed = cv2.morphologyEx(diff_opened, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours based on area
    contours, _ = cv2.findContours(diff_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(diff_closed)
    min_area = 50

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(result, [contour], -1, 255, -1)

    return result


def convert_to_svg(input_path, output_path, args):
    """
    Convert image to SVG with customizable parameters.

    Args:
        input_path (str): Input image path
        output_path (str): Output SVG path
        args (argparse.Namespace): Parsed command-line arguments
    """
    try:
        vtracer.convert_image_to_svg_py(
            input_path,
            output_path,
            colormode=args.colormode,
            mode=args.mode,
            filter_speckle=args.filter_speckle,
            color_precision=args.color_precision,
            corner_threshold=args.corner_threshold,
            length_threshold=args.length_threshold,
            splice_threshold=args.splice_threshold,
            path_precision=args.path_precision
        )
        print(f"Successfully converted {output_path}")
    except Exception as e:
        print(f"Error converting {output_path}: {e}")


def merge_svgs(output_dir):
    cell1_svg_path = os.path.join(output_dir, 'cell_1.svg')
    with open(cell1_svg_path, 'r', encoding='utf-8') as f:
        base_svg = f.read()


    end_tag_pos = base_svg.find('</svg>')
    if end_tag_pos == -1:
        print("Unable to find end tag in base SVG")
        return


    merged_content = base_svg[:end_tag_pos]

    # Process difference files
    for i in range(1, 4):
        diff_file = os.path.join(output_dir, f'difference_{i}.svg')
        try:
            with open(diff_file, 'r', encoding='utf-8') as f:
                diff_content = f.read()


            path_start = diff_content.find('<path')
            if path_start == -1:
                path_start = diff_content.find('<g')

            if path_start == -1:
                print(f"No path data found in {diff_file}")
                continue

            path_end = diff_content.find('</svg>')
            if path_end == -1:
                print(f"No end tag found in {diff_file}")
                continue

            path_content = diff_content[path_start:path_end]
            merged_content += '\n' + path_content

            print(f"Added content from {diff_file}")

        except FileNotFoundError:
            print(f"File not found: {diff_file}")
        except Exception as e:
            print(f"Error processing {diff_file}: {e}")

    merged_content += '\n</svg>'

    # 保存合并后的SVG文件
    try:
        merged_file_path = os.path.join(output_dir, 'merged_result.svg')
        with open(merged_file_path, 'w', encoding='utf-8') as f:
            f.write(merged_content)
        print(f"SVG merge completed, saved as {merged_file_path}")
    except Exception as e:
        print(f"Error saving merged file: {e}")


def main():
    # Parse command-line arguments
    args = parse_arguments()

    input_image = args.input

    output_dir = args.output if args.output else None

    cells, output_dir = split_image_to_4(input_image,output_dir)

    convert_to_svg(os.path.join(output_dir, 'cell_1.png'),
                   os.path.join(output_dir, 'cell_1.svg'),args)
    convert_to_svg(os.path.join(output_dir, 'cell_2.png'),
                   os.path.join(output_dir, 'cell_2.svg'),args)

    # Process consecutive image pairs
    processed_images = []
    for i in range(1, 4):
        result_path, svg_path = process_consecutive_images(
            os.path.join(output_dir, f'cell_{i}.png'),
            os.path.join(output_dir, f'cell_{i + 1}.png'),
            i,
            output_dir,
            thresh=args.thresh
        )
        if result_path and svg_path:
            convert_to_svg(result_path, svg_path, args)
            processed_images.append(svg_path)

    merge_svgs(output_dir)


if __name__ == "__main__":
    main()


