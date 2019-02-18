using Microsoft.ML.Data;

namespace ImageRecognitionOnnxSample.Data
{
    public class MobileNetOnnxPrediction
    {
        [ColumnName(MobileNetOnnxClassification.OutputName)]
        public float [] Output { get; set; }
    }
}
