using Microsoft.ML.Data;

namespace ImageRecognitionOnnxSample.Data
{
    public class MobileNetOnnxImageVector
    {
        [Column("R")]
        //[Column("G")]
        //[Column("B")]
        [VectorType(224,224,3)]
        public float [] ImagePixels { get; set; }
    }
}
