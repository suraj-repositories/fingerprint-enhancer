import cv2
import numpy as np
from pathlib import Path
from fingerprint_enhancer.fingerprint_image_enhancer import FingerprintImageEnhancer

if __name__ == "__main__":
    image_enhancer = FingerprintImageEnhancer()

    img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("input.png")

    output_dir = Path("enhanced")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "output.png"
    sharped_path = output_dir / "sharped-output.png"

    image_enhancer.enhance(img, invert_output=True)
    image_enhancer.save_enhanced_image(str(output_path))

    enhanced = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)

    enhanced = cv2.resize(
        enhanced,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    enhanced = cv2.addWeighted(enhanced, 1.8, blur, -0.8, 0)

    _, enhanced = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite(str(sharped_path), enhanced)

    print(f"Saved: {output_path}")
    print(f"Saved: {sharped_path}")