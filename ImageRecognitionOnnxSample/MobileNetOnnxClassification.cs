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
    public class MobileNetOnnxClassification
    {
        public const string OutputName = "mobilenetv20_output_flatten0_reshape0";
        private const int ImageHeight = 224;
        private const int ImageWidth = 224;

        private MLContext _mlContext;
        private string _modelFilePath;
        private List<string> _labels;

        public MobileNetOnnxClassification(MLContext mlContext, string modelFilePath, List<string> labels)
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
                    .Append(_mlContext.Transforms.Concatenate("data", "Red", "Green", "Blue"))
                    .Append(new OnnxScoringEstimator(_mlContext, new string[] { @"mobilenetv20_output_flatten0_reshape0" }, new string[] { "data" }, _modelFilePath))
                    .Append(new CustomMappingEstimator<MobileNetOnnxPrediction, ImagePrediction>(_mlContext, contractName: "MobileNetExtractor",
                          mapAction: (networkResult, prediction) =>
                          {
                              prediction.Estimate = networkResult.Output.Max();
                              prediction.Index = networkResult.Output.ToList().IndexOf(prediction.Estimate);
                              prediction.Label = _labels[prediction.Index];
                          }));

            //attention code below doesn't work correctly ¯\_(ツ)_/¯
            //var pipeline = new ImageLoadingEstimator(_mlContext, string.Empty, ("ImageData", "ImagePath"))
            //        .Append(new ImageResizingEstimator(_mlContext, "ImageResized", ImageWidth, ImageHeight, "ImageData"))
            //        .Append(new ImagePixelExtractingEstimator(_mlContext, "Red", "ImageResized", colors: ColorBits.Red, offset: 0.485f * 255, scale: 1 / (0.229f * 255)))
            //        .Append(new ImagePixelExtractingEstimator(_mlContext, "Green", "ImageResized", colors: ColorBits.Green, offset: 0.456f * 255, scale: 1 / (0.224f * 255)))
            //        .Append(new ImagePixelExtractingEstimator(_mlContext, "Blue", "ImageResized", colors: ColorBits.Blue, offset: 0.406f * 255, scale: 1 / (0.225f * 255)))
            //        .Append(new ColumnConcatenatingEstimator(_mlContext, "data", "Red", "Green", "Blue"))
            //        .Append(new OnnxScoringEstimator(_mlContext, new string[] { @"mobilenetv20_output_flatten0_reshape0" }, new string[] { "data" }, _modelFilePath))
            //        .Append(new CustomMappingEstimator<MobileNetOnnxPrediction, ImagePrediction>(_mlContext, contractName: "MobileNetExtractor",
            //          mapAction: (networkResult, prediction) =>
            //          {
            //              prediction.Estimate = networkResult.Output.Max();
            //              prediction.Index = networkResult.Output.ToList().IndexOf(prediction.Estimate);
            //              prediction.Label = _labels[prediction.Index];
            //          }));

            var transformer = pipeline.Fit(data);

            return transformer.CreatePredictionEngine<ImageData, ImagePrediction>(_mlContext);
        }
    }
}
