using ImageRecognitionOnnxSample.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Normalizers;
using Microsoft.ML.Transforms.Projections;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ImageRecognitionOnnxSample
{
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
              .Append(new ImageResizingEstimator(_mlContext, "ImageResized", ImageWidth, ImageHeight, "ImageData"))
              .Append(new ImagePixelExtractingEstimator(_mlContext, "data_0", "ImageResized", interleave: true/*, scale: (float)(1.0 / 255)*/))
              //Append(new NormalizingEstimator(_mlContext, "nrm0_1", "ImagePixels", mode: NormalizingEstimator.NormalizerMode.SupervisedBinning))
              //.Append(new LpNormalizingEstimator(_mlContext, new[] {("", ""), ("", "")}))
              .Append(new OnnxScoringEstimator(_mlContext, new string[] { OutputName }, new string[] { "data_0" }, _modelFilePath))
              .Append(new CustomMappingEstimator<SqueezeNetOnnxPrediction, ImagePrediction>(_mlContext, contractName: "MobileNetExtractor",
                    mapAction: (networkResult, prediction) =>
                    {
                        prediction.Estimate = networkResult.Output.Max();
                        prediction.Index = networkResult.Output.ToList().IndexOf(prediction.Estimate);
                        prediction.Label = _labels[prediction.Index];
                    }));

            var transformer = pipeline.Fit(data);

            //var transformedData = transformer.Transform(data);
            //var pixels = transformedData.GetColumn<VBuffer<float>>(_mlContext, "data_0");

            return transformer.CreatePredictionEngine<ImageData, ImagePrediction> (_mlContext);
        }
    }
}
