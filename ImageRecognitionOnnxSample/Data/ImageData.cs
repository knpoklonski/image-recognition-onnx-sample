using Microsoft.ML.Data;

namespace ImageRecognitionOnnxSample.Data
{
    public class ImageData
    {
        [Column("0", "ImagePath")]
        public string ImagePath { get; set; }
    }
}
