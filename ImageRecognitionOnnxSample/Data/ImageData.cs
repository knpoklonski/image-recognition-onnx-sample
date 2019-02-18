using Microsoft.ML.Data;

namespace ImageRecognitionOnnxSample.Data
{
    public class ImageData
    {
        [Column("0", "ImagePath")]
        public string ImagePath { get; set; }

        //[Column("1", "mean")]
        //[VectorType(1)]
        //public float[] Mean {
        //    get {
        //        return new[] { 0.485f, 0.456f, 0.406f };
        //    }
        //}

        //[Column("1", "std")]
        //[VectorType(1)]
        //public float [] Std
        //{
        //    get
        //    {
        //        return new[] { 0.229f, 0.224f, 0.225f };
        //    }
        //}
    }
}
