using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ImageRecognitionOnnxSample
{
    public static class Helpers
    {
        public static Action<string, IEnumerable<VBuffer<float>>> print = (colName, column) =>
        {
            Console.WriteLine($"{colName} column obtained post-transformation.");
            foreach (var row in column)
                Console.WriteLine($"{string.Join(" ", row.DenseValues().Select(x => x.ToString()))} ");
        };
    }
}
