# Real time FaceMask Detect

![Overall](https://github.com/zjsjayce/FaceMaskDetect/blob/main/Architecture.jpeg)


- [Train Model](https://colab.research.google.com/drive/1q1_Xji4wg2pDYHeJx_m75mSZYLeUXK_q?usp=sharing) is the first step in this project, you could find it in the colab or in the ModelTrain folder. 

- Then you need to output your the model or you could use my model at [here](https://drive.google.com/drive/folders/1jKv8Vnbv-os5Ab9D2FCE4p36-tPFXId5?usp=sharing). 

- Next you need to convert the .pt into CoreML, the [source code](https://colab.research.google.com/drive/1er09xThb4TFp_yuhCLVXmiYi-XTNnUPC?usp=sharing) you could also find in ModelTrain folder.

- [Create ML](https://developer.apple.com/documentation/create_ml) enable us to train [Core ML](https://developer.apple.com/documentation/coreml) models just with Drag-and-Drop. Then, `MLModelCamera` enable to test the `.mlmodel` files as a real-time `Image Classiffication` or `Object Detection` app just with Drag-and-Drop.

## Usage

- Put your `.mlmodel` files into the `Application/models` folder.

That's it! You don't need to add the models manually to the project.

After running the app on your iOS device, you can choose the model with the "Change" button.


## Supporting Model Types

- Image Classification
- Object Detection


## Requirements

### The models are created with Create ML

- Xcode 12+
- iOS 12+

### The models are created with coremltools

- Xcode 12+
- iOS 12+

## How to find trained models

- I used the Yolov5s as the model to detect the mask wearing status.

## How to find trained models

- If you want to run at your own device, you need to adjust the developer account the siging certificate.
## The presentation
- - [8_15 Final Presentation.pptx](https://northeastern-my.sharepoint.com/:p:/r/personal/xue_haow_northeastern_edu/Documents/Capstone Project Summer/8_15 Final Presentation.pptx?d=wdf3583b9cb8f4331a66ce05cd867b9f0&csf=1&web=1&e=iKGaLA)


## Author

**[Jayce Zhang](https://www.linkedin.com/in/jaycezhang/)**

