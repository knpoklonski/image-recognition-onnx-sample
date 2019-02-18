using Microsoft.ML.Data;

namespace ImageRecognitionOnnxSample.Data
{
    public class SqueezeNetOnnxPrediction
    {
        [ColumnName(SqueezeNetOnnxClassification.OutputName)]
        public float [] Output { get; set; }
    }
}
