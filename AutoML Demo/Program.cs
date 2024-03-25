using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.AccessControl;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace AutoML_Demo
{
    public class Program
    {
        static void Main(string[] args)
        {
            PickTrainOrTestInConsole();
        }

        public static void PickTrainOrTestInConsole()
        {
            Console.WriteLine("\n(1) Train model");
            Console.WriteLine("(2) Test model");
            Console.WriteLine("(3) Exit");
            Console.WriteLine("\nEnter your choice:");

            var choice = Console.ReadLine();

            switch (choice)
            {
                case "1":
                    Console.WriteLine("\nEnter the maximum time to train the model (in seconds):");
                    var time = Console.ReadLine();
                    if (uint.TryParse(time, out var maxExperimentTimeInSeconds))
                    {
                        TrainModel.Train(maxExperimentTimeInSeconds);
                    }
                    else
                    { 
                        Console.WriteLine("Invalid time");
                    }
                    break;
                case "2":
                    TestModel.Test();
                    break;
                case "3":
                    Environment.Exit(0);
                    break;
                default:
                    Console.WriteLine("Invalid choice");
                    break;
            }

            PickTrainOrTestInConsole();
        }
    }
}
