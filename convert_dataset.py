import json
import os
import shutil

# =========================
# 기본 경로
# =========================
TRAIN_JSON = "archive/mvtec_screws_train.json"
VAL_JSON = "archive/mvtec_screws_val.json"
IMAGE_DIR = "archive/images"

OUTPUT_DIR = "dataset"

# =========================
# dataset 폴더 만들기
# =========================
for folder in [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val"
]:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)


# =========================
# JSON → YOLO 변환 함수
# =========================
def convert(json_path, split):

    print(f"\n===== {split} 데이터 변환 시작 =====")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 이미지 정보
    images = {
        img["id"]: img
        for img in data["images"]
    }

    count = 0

    for ann in data["annotations"]:

        image_id = ann["image_id"]

        if image_id not in images:
            continue

        image_info = images[image_id]

        file_name = os.path.basename(image_info["file_name"])

        src_image = os.path.join(
            IMAGE_DIR,
            file_name
        )

        if not os.path.exists(src_image):
            print("이미지 없음:", src_image)
            continue

        # 이미지 복사
        dst_image = os.path.join(
            OUTPUT_DIR,
            "images",
            split,
            file_name
        )

        if not os.path.exists(dst_image):
            shutil.copy2(src_image, dst_image)

        # =========================
        # bbox
        # =========================
        bbox = ann["bbox"]

        # MVTec 형식:
        # row, col, width, height, phi
        row = float(bbox[0])
        col = float(bbox[1])
        width = float(bbox[2])
        height = float(bbox[3])

        image_width = image_info["width"]
        image_height = image_info["height"]

        # YOLO 형식
        x_center = col / image_width
        y_center = row / image_height

        box_width = width / image_width
        box_height = height / image_height

        # category_id는 1부터 시작
        # YOLO class는 0부터 시작
        class_id = ann["category_id"] - 1

        label_file = os.path.splitext(file_name)[0] + ".txt"

        label_path = os.path.join(
            OUTPUT_DIR,
            "labels",
            split,
            label_file
        )

        with open(label_path, "a", encoding="utf-8") as f:
            f.write(
                f"{class_id} "
                f"{x_center} "
                f"{y_center} "
                f"{box_width} "
                f"{box_height}\n"
            )

        count += 1

    print(f"{split} 변환 완료")
    print(f"annotation 수: {count}")


# =========================
# Train / Val 변환
# =========================
convert(TRAIN_JSON, "train")
convert(VAL_JSON, "val")


# =========================
# data.yaml 생성
# =========================
yaml_path = os.path.join(
    OUTPUT_DIR,
    "data.yaml"
)

with open(yaml_path, "w", encoding="utf-8") as f:

    f.write("path: ./dataset\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")

    f.write("names:\n")

    for i in range(1, 14):
        f.write(f"  {i - 1}: type_{i:03d}\n")


print("\n==============================")
print("✅ 데이터셋 생성 완료!")
print("==============================")
print("dataset 폴더를 확인하세요.")