using Microsoft.ML.AutoML;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AutoML_Demo
{
    public class TrainModel
    {
        /// <summary>
        /// Trains a model using the specified maximum experiment time and saves the model to a file
        /// </summary>
        /// <param name="maxExperimentTimeInSeconds">The maximum time to train the model in seconds</param>
        public static void Train(uint maxExperimentTimeInSeconds)
        {
            try
            {
                Console.WriteLine("\n===== Training Model =====");
                Console.WriteLine("This may take a while...");

                var mlContext = new MLContext();

                // Step 1: Load the data
                var trainingData = mlContext.Data.LoadFromTextFile<ModelInput>(
                    path: "Data/optdigits-train.csv",
                    hasHeader: false,
                    separatorChar: ','
                );

                // Step 2: Define the experiment
                var experimentSettings = new MulticlassExperimentSettings
                {
                    MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds,
                    OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy
                };
                var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(experimentSettings);

                // Step 3: Run the experiment using the training data
                var result = experiment.Execute(trainingData, labelColumnName: "Number");

                // Step 4: Get the best performing model from the experiment
                var model = result.BestRun.Model;

                // Step 5: Save the model
                mlContext.Model.Save(model, trainingData.Schema, "Model.zip");

                Console.WriteLine("\n===== Training Complete =====");
                Console.WriteLine("The model has been saved as Model.zip");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nAn error occurred: {ex.Message}");
            }
        }
    }
}
