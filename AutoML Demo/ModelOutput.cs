using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AutoML_Demo
{
    public class ModelOutput
    {
        [ColumnName("Score")]
        public float[] Score { get; set; }
    }
}
