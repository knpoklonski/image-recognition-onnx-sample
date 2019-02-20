using ImageRecognitionOnnxSample.Data;
using System;
using System.IO;
using System.Linq;

namespace ImageRecognitionOnnxSample
{
    class Program
    {
        static void Main(string[] args)
        {
            var imagePath = Path.Combine(Directory.GetCurrentDirectory(), args.Length > 0 ? args[0] : "images/panda.jpg");
            var labels = File.ReadLines(Path.Combine(Directory.GetCurrentDirectory(), "labels.txt")).ToList();

            // Result for Tensorflow MobileNet model
            var tensorFlowModelPath = Path.Combine(Directory.GetCurrentDirectory(), "mobilenet_v2_1.4_224_frozen.pb");
            var mobileNetTensorFlow = new MobileNetTensorflowClassification(new Microsoft.ML.MLContext(), tensorFlowModelPath, labels);
            var imageClassifier = mobileNetTensorFlow.CreateClassifier();
            var prediction = imageClassifier.Predict(new ImageData { ImagePath = imagePath });
            Console.WriteLine("MobileNet tensorflow model top prediction {0} {1}, with estimate {2}", prediction.Index, prediction.Label, prediction.Estimate);

            // Result for ONNX MobileNet model
            var onnxModelPath = Path.Combine(Directory.GetCurrentDirectory(), "mobilenetv2-1.0.onnx");
            var mobileNetOnnx = new MobileNetOnnxClassification(new Microsoft.ML.MLContext(), onnxModelPath, labels);
            var imageOnnxClassifier = mobileNetOnnx.CreateClassifier();
            var predictionOnnx = imageOnnxClassifier.Predict(new ImageData { ImagePath = imagePath });
            Console.WriteLine("MobileNet onnx model top prediction {0} {1}, with estimate {2}", predictionOnnx.Index, predictionOnnx.Label, predictionOnnx.Estimate);

            //Results for ONNX Squeeze model
            var onnxSqueezeNetModelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            var squeezeNetOnnx = new SqueezeNetOnnxClassification(new Microsoft.ML.MLContext(), onnxSqueezeNetModelPath, labels);
            var squeezeNetOnnxClassifier = squeezeNetOnnx.CreateClassifier();
            var predictionSqueezeNet = squeezeNetOnnxClassifier.Predict(new ImageData { ImagePath = imagePath });
            Console.WriteLine("SqueezeNet onnx model top prediction {0} {1}, with estimate {2}", predictionSqueezeNet.Index, predictionSqueezeNet.Label, predictionSqueezeNet.Estimate);

            Console.ReadKey();
        }
    }
}
