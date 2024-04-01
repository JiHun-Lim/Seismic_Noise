from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import sympy                                                                    

def add_white_background(image_path, output_path):
    # 이미지 열기
    image = Image.open(image_path)

    image = image.crop((20, 0,610, 320))
    
    # 이미지 크기 가져오기
    width, height = image.size
    
    # 흰색 배경 이미지 생성
    background = Image.new('RGB', (700, 420), color='white')
    
    # 원본 이미지를 흰색 배경 이미지 왼쪽에 복사
    background.paste(image, (50, 50))

    draw = ImageDraw.Draw(image)

    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=25)
    # font = ImageFont.load_default(font_size=font_size)

    background_draw = ImageDraw.Draw(background)
    background_draw.text((280, 380), "Period [s]", fill="black", font=font)
    background_draw.text((135, 120), "NHNM", fill="black", font=font)
    background_draw.text((135, 270), "NLNM", fill="black", font=font)

    # 결과 이미지 저장
    background.save(output_path)

def add_caption(image_path, output_path):
    # 이미지 열기
    image = Image.open(image_path)
    rotate_image = image.rotate(270, expand=True)

    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=25)
    # font = ImageFont.load_default(font_size=font_size)

    x = sympy.symbols('x')                                                          
    y = 1 + sympy.sin(sympy.sqrt(x**2 + 20))                                         
    lat = sympy.latex(y) 

    rotate_image_draw = ImageDraw.Draw(rotate_image)
    # rotate_image_draw.text((45, 10), "Amplitude [m2/s4/Hz] [dB]", fill="black", font=font)
    rotate_image_draw.text((120, 10), "Amplitude [dB]", fill="black", font=font)
    rotate_image_draw.text((200, 650), "[%]", fill="black", font=font)
    rotate_image = rotate_image.rotate(90, expand=True)

    rotate_image.save(output_path)

# 이미지 경로와 결과 이미지 경로를 지정합니다.
image_path = "real_chc2.png"
background_path = "output_with_white_background.png"
final_path = image_path[:-4] + "_after.png"

# 함수를 호출하여 이미지 왼쪽에 흰색 빈 배경을 추가합니다.
add_white_background(image_path, background_path)
add_caption(background_path, final_path)

