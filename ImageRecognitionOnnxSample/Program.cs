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
            var imagePath = Path.Combine(Directory.GetCurrentDirectory(), args.Length > 0 ? args[0] : "panda.jpg");
            var labels = File.ReadLines(Path.Combine(Directory.GetCurrentDirectory(), "labels.txt")).ToList();

            var tensorFlowModelPath = Path.Combine(Directory.GetCurrentDirectory(), "mobilenet_v2_1.4_224_frozen.pb");
            var mobileNetTensorFlow = new MobileNetTensorflowClassification(new Microsoft.ML.MLContext(), tensorFlowModelPath, labels);
            var imageClassifier = mobileNetTensorFlow.CreateClassifier();
            var prediction = imageClassifier.Predict(new ImageData { ImagePath = imagePath });
            Console.WriteLine("MobileNet tensorflow model top prediction {0} {1}, with estimate {2}", prediction.Index, prediction.Label, prediction.Estimate);
        }
    }
}
