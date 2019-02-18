using Microsoft.ML.Data;

namespace ImageRecognitionOnnxSample.Data
{
    public class MovileNetTensorflowPrediction
    {
        [ColumnName(MobileNetTensorflowClassification.OutputName)]
        public float [] Output { get; set; }
    }
}
