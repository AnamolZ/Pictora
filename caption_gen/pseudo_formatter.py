import csv
from collections import defaultdict

input_file = "pseudo_caption/pseudo_caption.txt"
output_file = "../training_data/pseudo_caption/pseudo_caption.txt"

captions_by_image = defaultdict(list)

with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        if "#0\t" in line:
            parts = line.strip().split("#0\t", 1)
            if len(parts) == 2:
                image_id, caption = parts
                captions_by_image[image_id.strip()].append(caption.strip())

with open(output_file, "w", encoding="utf-8", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["image", "caption"])
    for image_id, captions in captions_by_image.items():
        for caption in captions:
            writer.writerow([image_id, f" {caption}"])
