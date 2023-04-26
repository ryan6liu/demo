import pytesseract
from PIL import Image

img_path = 'tesseract\\pic\\Image_20230421094331.jpg'

text = pytesseract.image_to_string(Image.open(img_path), lang='chi_sim')
# text = pytesseract.image_to_string(Image.open(img_path))
print(text)

