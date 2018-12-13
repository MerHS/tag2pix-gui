# Dependencies
* Python 3.6+
* Pytorch 0.4
* numpy
* Pillow
* scikit-image

# How to Colorize 

1. Load Sketch -> 스케치 로드
2. Simplify Sketch -> 스케치 단순화 (생략 가능 / 바로 colorize 가능함)
    * 우상단의 Generated Size에서 단순화 출력물의 사이즈를 설정할 수 있습니다. (권장: 768)
3. 우측의 태그 리스트중에 원하는 것들을 다중 선택
4. Colorize -> 채색 진행 (출력: 256x256)
5. Upscale -> Waifu2x를 이용하여 결과물 해상도 증강 (생략 가능)
6. Save -> 파일 저장

# LICENSE

https://github.com/bobbens/sketch_simplification 의 model_gan.t7을 이용하여 sketch simplification을 진행하기 때문에 LICENSE 문제로 비상업적 내부 이용만 가능합니다.

https://github.com/yu45020/Waifu2x 또한 사용하여 GPL v3을 따릅니다. 