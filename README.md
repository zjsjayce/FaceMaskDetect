# FaceMaskDetect

[Create ML](https://developer.apple.com/documentation/create_ml) enable us to train [Core ML](https://developer.apple.com/documentation/coreml) models just with Drag-and-Drop. Then, `MLModelCamera` enable to test the `.mlmodel` files as a real-time `Image Classiffication` or `Object Detection` app just with Drag-and-Drop.

## Usage

- Put your `.mlmodel` files into the `/models` folder.

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

## Author

**[Jayce Zhang]()**

