using Microsoft.ML.Data;

namespace ImageRecognitionOnnxSample.Data
{
    public class MobileNetOnnxImageRGB
    {
        [VectorType(224, 224)]
        public float [] Red { get; set; }
        [VectorType(224, 224)]
        public float [] Green { get; set; }
        [VectorType(224, 224)]
        public float [] Blue { get; set; }
    }
}
