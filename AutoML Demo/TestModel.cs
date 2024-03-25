using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AutoML_Demo
{
    public class TestModel
    {
        /// <summary>
        /// Tests the model using the test data and outputs the evaluation metrics
        /// </summary>
        public static void Test()
        {
            var mlContext = new MLContext();

            // Step 1: Load the model
            ITransformer model = mlContext.Model.Load("Model.zip", out DataViewSchema schema);

            // Step 2: Load the test data
            var testData = mlContext.Data.LoadFromTextFile<ModelInput>(
                path: "Data/optdigits-test.csv",
                hasHeader: false,
                separatorChar: ','
            );

            // Step 3: Make predictions
            var predictions = model.Transform(testData);

            // Step 4: Evaluate the model
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Number", scoreColumnName: "Score");

            // Step 5: Output the evaluation metrics
            Console.WriteLine($"\nAccuracy: {metrics.MicroAccuracy * 100:F2}%");
            Console.WriteLine($"Log Loss: {metrics.LogLoss:F4}");
            Console.WriteLine($"{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");
        }
    }
}
