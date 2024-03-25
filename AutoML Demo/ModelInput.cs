using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AutoML_Demo
{
    public class ModelInput
    {
        [LoadColumn(0, 63), VectorType(64)]
        public float[] PixelValues { get; set; }

        [LoadColumn(64)]
        public float Number { get; set; }
    }
}
