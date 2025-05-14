def parse_pseudo_caption_to_csv(input_txt, output_csv):
    try:
        with open(input_txt, 'r', encoding='utf-8') as txt_file, open(output_csv, 'w', encoding='utf-8', newline='') as csv_file:
            csv_file.write("image,caption\n")
            next(txt_file)
            for line in txt_file:
                line = line.strip()
                if line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        image_part = parts[0].strip()
                        image_name = image_part.split()[-1]
                        caption = parts[1].strip()
                        csv_file.write(f"{image_name}, {caption}\n")
                    else:
                        print(f"Skipping invalid line: {line}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    input_file = "../train_model/training_data/pseudo_caption/pseudo_caption.csv"
    output_file = "../train_model/training_data/pseudo_caption/pseudo_caption.txt"
    parse_pseudo_caption_to_csv(input_file, output_file)