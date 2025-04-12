import string

class DescriptionProcessor:
    def load_doc(self, filename):
        with open(filename, 'r') as file:
            return file.read()

    def all_img_captions(self, filename):
        text = self.load_doc(filename)
        captions = text.strip().split('\n')
        descriptions = {}
        for caption in captions:
            img, cap = caption.split('\t')
            img = img.split('#')[0]
            if img not in descriptions:
                descriptions[img] = []
            descriptions[img].append(cap)
        return descriptions

    def cleaning_text(self, captions):
        table = str.maketrans('', '', string.punctuation)
        for img, caps in captions.items():
            for i, cap in enumerate(caps):
                cap = cap.replace("-", " ")
                desc = cap.split()
                desc = [word.lower() for word in desc]
                desc = [word.translate(table) for word in desc]
                desc = [word for word in desc if len(word) > 1]
                desc = [word for word in desc if word.isalpha()]
                captions[img][i] = ' '.join(desc)
        return captions

    def save_descriptions(self, descriptions, filename):
        lines = []
        for key, caps in descriptions.items():
            for cap in caps:
                lines.append(key + '\t' + cap)
        data = "\n".join(lines)
        with open(filename, "w") as file:
            file.write(data)

    def load_clean_descriptions(self, filename, photos):
        text = self.load_doc(filename)
        descriptions = {}
        for line in text.strip().split('\n'):
            tokens = line.split()
            if len(tokens) < 1:
                continue
            image, image_caption = tokens[0], tokens[1:]
            if image in photos:
                if image not in descriptions:
                    descriptions[image] = []
                desc = '<start> ' + " ".join(image_caption) + ' <end>'
                descriptions[image].append(desc)
        return descriptions
