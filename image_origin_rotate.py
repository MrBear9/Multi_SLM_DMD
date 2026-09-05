
import cv2
import numpy as np
import os
from pathlib import Path


def rotate_images(input_dir, output_dir, angle=45):
    """
    将输入目录下的所有图片旋转指定角度
    :param input_dir: 输入图片目录路径
    :param output_dir: 输出目录路径
    :param angle: 旋转角度（度），正值为逆时针旋转
    """
    os.makedirs(output_dir, exist_ok=True)

    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir()
                   if f.is_file() and f.suffix.lower() in supported_formats]

    if not image_files:
        print(f"在目录 {input_dir} 中未找到支持的图片文件")
        return

    print(f"找到 {len(image_files)} 个图片文件需要处理")

    processed_count = 0
    for image_file in image_files:
        try:
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"无法读取图片: {image_file}")
                continue

            h, w = img.shape[:2]
            center = (w // 2, h // 2)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            cos_val = np.abs(rotation_matrix[0, 0])
            sin_val = np.abs(rotation_matrix[0, 1])

            new_w = int((h * sin_val) + (w * cos_val))
            new_h = int((h * cos_val) + (w * sin_val))

            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]

            rotated_img = cv2.warpAffine(img, rotation_matrix, (new_w, new_h),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))

            output_filename = f"{image_file.stem}_rotated{image_file.suffix}"
            output_path = os.path.join(output_dir, output_filename)

            cv2.imwrite(output_path, rotated_img)
            processed_count += 1
            print(f"处理完成: {image_file.name} -> {output_filename} "
                  f"(旋转角度: {angle}°, 原始尺寸: {w}×{h}, 新尺寸: {new_w}×{new_h})")

        except Exception as e:
            print(f"处理图片 {image_file.name} 时出错: {e}")

    print(f"\n处理完成！共处理 {processed_count}/{len(image_files)} 个图片文件")
    print(f"输出目录: {output_dir}")


def batch_rotate_images(angle=45):
    """
    批量处理图片：将image_Origin_size目录下的图片旋转指定角度并保存到image_Origin_rotated目录
    :param angle: 旋转角度（度），正值为逆时针旋转
    """
    base_dir = Path(__file__).parent
    input_dir = base_dir / "slm2"
    output_dir = base_dir / "slm2_rotated"

    if not input_dir.exists():
        print(f"错误：输入目录不存在: {input_dir}")
        return

    print(f"开始批量旋转图片...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"旋转角度: {angle}°")
    print("-" * 50)

    rotate_images(str(input_dir), str(output_dir), angle)


if __name__ == "__main__":
    batch_rotate_images(90)