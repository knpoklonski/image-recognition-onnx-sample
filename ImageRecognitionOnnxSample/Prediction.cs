using Microsoft.ML.Data;

namespace ImageRecognitionOnnxSample
{
    public class Prediction
    {
        [ColumnName(@"MobilenetV2/Predictions/Reshape_1")]
        [VectorType(1000)]
        public float[] Probabilities { get; set; }
    }
}
