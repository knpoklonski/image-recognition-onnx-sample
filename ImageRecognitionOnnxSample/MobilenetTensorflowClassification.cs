using ImageRecognitionOnnxSample.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Transforms.Projections;

namespace ImageRecognitionOnnxSample
{

    public class MobileNetTensorflowClassification
    {
        public const string OutputName = "MobilenetV2/Predictions/Reshape_1";
        private const int ImageHeight = 224;
        private const int ImageWidth = 224;

        private MLContext _mlContext;
        private string _modelFilePath;
        private List<string> _labels;

        public MobileNetTensorflowClassification(MLContext mlContext, string modelFilePath, List<string> labels)
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
              .Append(new ImagePixelExtractingEstimator(_mlContext, "input", "ImageResized", colors: ImagePixelExtractorTransformer.ColorBits.Rgb, interleave:true, asFloat: true, offset: 128f, scale: 1/255f))
              .Append(new TensorFlowEstimator(_mlContext, new string[] { @"MobilenetV2/Predictions/Reshape_1" }, new string[] { "input" }, _modelFilePath))
              .Append(new CustomMappingEstimator<MovileNetTensorflowPrediction, ImagePrediction>(_mlContext, contractName: "MobileNetExtractor",
                    mapAction: (networkResult, prediction) =>
                    {
                        prediction.Estimate = networkResult.Output.Max();
                        prediction.Index = networkResult.Output.ToList().IndexOf(prediction.Estimate) - 1; //-1 because result contains 1001
                        prediction.Label = _labels[prediction.Index];
                    }));

            var transformer = pipeline.Fit(data);

            return transformer.CreatePredictionEngine<ImageData, ImagePrediction> (_mlContext);
        }
    }
}
