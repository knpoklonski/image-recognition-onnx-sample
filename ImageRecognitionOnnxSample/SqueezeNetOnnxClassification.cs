using ImageRecognitionOnnxSample.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using static Microsoft.ML.ImageAnalytics.ImagePixelExtractorTransformer;

namespace ImageRecognitionOnnxSample
{
    //ToDo check results
    public class SqueezeNetOnnxClassification
    {
        public const string OutputName = "softmaxout_1";
        private const int ImageHeight = 224;
        private const int ImageWidth = 224;
        
        private MLContext _mlContext;
        private string _modelFilePath;
        private List<string> _labels;

        public SqueezeNetOnnxClassification(MLContext mlContext, string modelFilePath, List<string> labels)
        {
            _mlContext = mlContext;
            _modelFilePath = modelFilePath;
            _labels = labels;
        }

        public PredictionEngine<ImageData, ImagePrediction> CreateClassifier()
        {
            var data = _mlContext.Data.ReadFromEnumerable(new List<ImageData>());

            var pipeline = new ImageLoadingEstimator(_mlContext, string.Empty, ("ImageData", "ImagePath"))
                    .Append(_mlContext.Transforms.Resize("ImageResized", imageWidth: ImageWidth, imageHeight: ImageHeight, inputColumnName: "ImageData"))
                    .Append(_mlContext.Transforms.ExtractPixels(new ColumnInfo("Red", "ImageResized",
                            colors: ColorBits.Red, offset: 0.485f * 255, scale: 1 / (0.229f * 255))))
                    .Append(_mlContext.Transforms.ExtractPixels(new ColumnInfo("Green", "ImageResized",
                            colors: ColorBits.Green, offset: 0.456f * 255, scale: 1 / (0.224f * 255))))
                    .Append(_mlContext.Transforms.ExtractPixels(new ColumnInfo("Blue", "ImageResized",
                            colors: ColorBits.Blue, offset: 0.406f * 255, scale: 1 / (0.225f * 255))))
                    .Append(_mlContext.Transforms.Concatenate("data_0", "Red", "Green", "Blue"))
              .Append(new OnnxScoringEstimator(_mlContext, new string[] { OutputName }, new string[] { "data_0" }, _modelFilePath))
              .Append(new CustomMappingEstimator<SqueezeNetOnnxPrediction, ImagePrediction>(_mlContext, contractName: "MobileNetExtractor",
                    mapAction: (networkResult, prediction) =>
                    {
                        prediction.Estimate = networkResult.Output.Max();
                        prediction.Index = networkResult.Output.ToList().IndexOf(prediction.Estimate);
                        prediction.Label = _labels[prediction.Index];
                    }));

            var transformer = pipeline.Fit(data);

            return transformer.CreatePredictionEngine<ImageData, ImagePrediction> (_mlContext);
        }
    }
}
