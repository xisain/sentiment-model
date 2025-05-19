from datasets import load_dataset
import re

slang_dataset = load_dataset("zeroix07/indo-slang-words", split="train")
slang_dict = {}
for item in slang_dataset:
    if ':' in item['text']:
        slang, formal = item['text'].split(':', 1)
        slang_dict[slang.strip()] = formal.strip()
def text_cleansing(text):
    text = re.sub(r'([.,!?;:])(?=\S)', r'\1 ', text)  # tambahkan spasi setelah tanda baca umum
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)        # hapus simbol/tanda baca selain huruf/angka/spasi
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()          # hapus spasi berlebih
    return text

def normalize_slang(text, slang_dict):
    return ' '.join([slang_dict.get(word, word) for word in text.split()])