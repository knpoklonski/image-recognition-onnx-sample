using ImageRecognitionOnnxSample.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;

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
              .Append(new ImageResizingEstimator(_mlContext, "ImageResized", ImageWidth, ImageHeight, "ImageData"))
              .Append(new ImagePixelExtractingEstimator(_mlContext, "data", "ImageResized", interleave: true, asFloat: true, scale: 255f, offset: .1f))
              //.Append(new LpNormalizingEstimator(_mlContext, new[] { ("data", "mean"), ("data", "std")}))
              .Append(new OnnxScoringEstimator(_mlContext, new string[] { @"mobilenetv20_output_flatten0_reshape0" }, new string[] { "data" }, _modelFilePath))
              .Append(new CustomMappingEstimator<MobileNetOnnxPrediction, ImagePrediction>(_mlContext, contractName: "MobileNetExtractor",
                    mapAction: (networkResult, prediction) =>
                    {
                        prediction.Estimate = networkResult.Output.Max();
                        prediction.Index = networkResult.Output.ToList().IndexOf(prediction.Estimate);
                        prediction.Label = _labels[prediction.Index];
                    }));

            var transformer = pipeline.Fit(data);

            //var transformedData = transformer.Transform(data);
            //var pixels = transformedData.GetColumn<VBuffer<float>>(_mlContext, "data");
            //Helpers.print("data", pixels);

            return transformer.CreatePredictionEngine<ImageData, ImagePrediction> (_mlContext);
        }
    }
}
