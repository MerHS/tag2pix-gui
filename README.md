# Tag2Pix: Line Art Colorization using Text Tag with SECat and Changing Loss (ICCV 19')

## How to Run

1. Download Network Dumps from [tag2pix repo releases](https://github.com/MerHS/tag2pix)
2. (Optional) Download [waifu2x-caffe](https://example.org) and place it to waifu2x-caffe directory. (wafu2x-caffe.exe should be in this directory)
2. Install Dependencies
3. `python main.py`

## Dependencies

* Python 3.6+
* Pytorch 1.1.0
* numpy
* Pillow
* scikit-image

## Usage 

1. Load Sketch
2. 우측의 태그 리스트중에 원하는 것들을 다중 선택
3. Colorize -> 채색 진행 
4. Upscale (Windows only) -> waifu2x-caffe를 이용하여 결과물 해상도 증강 (생략 가능)
5. Save -> 파일 저장

# LICENSE

This program and network dumps can be used only for non-commercial, internal researches.

If you want to use it for commercial purpose, re-train the full tag2pix network with the methods of our paper.
(We cannot share exact train datasets due to the image licenses.)
