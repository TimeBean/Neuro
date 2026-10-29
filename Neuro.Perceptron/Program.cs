using System;
using System.IO;
using System.Linq;
using System.Text;
using CommandLine;
using Neuro.Functions;

namespace Neuro.Perceptron
{
    // ReSharper disable once ClassNeverInstantiated.Global
    internal class Program
    {
        private static readonly double[][] Inputs =
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ];

        private static readonly double[][] Targets =
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0]
        ];

        private static void Main(string[] args)
        {
            Parser.Default.ParseArguments<Options>(args)
                .WithParsed(RunOptions)
                .WithNotParsed(errors => { });
        }

        private static void RunOptions(Options opts)
        {
            IFunction sigmoid = new Sigmoid();
            var net = new Network.Network(
                learningRate: opts.LearningRate,
                activationFunction: sigmoid,
                opts.Layers.ToArray());

            Console.CursorVisible = false;
            Console.WriteLine($"Training for {opts.EpochCount} epochs...");

            for (var epoch = 0; epoch < opts.EpochCount; epoch++)
            {
                for (var i = 0; i < Inputs.Length; i++)
                {
                    net.Train(Inputs[i], Targets[i]);
                }

                DrawPipProgressBar(epoch + 1, opts.EpochCount);
            }

            Console.WriteLine("\n");
            Console.CursorVisible = true;

            var resultsText = GetTrainingResults(net);
            if (opts.WriteToFile)
            {
                var layersStr = string.Join("-", opts.Layers);

                if (!Directory.Exists("results"))
                {
                    Directory.CreateDirectory("results");
                }
                var filename = $"results/e{opts.EpochCount}l{layersStr}r{opts.LearningRate}.txt";

                File.WriteAllText(filename, resultsText);
                Console.WriteLine($"Results after training saved to {filename}.");
            }
            else
            {
                Console.Write(resultsText);
            }
        }

        private static void DrawPipProgressBar(int current, int total, int barLength = 30)
        {
            var progress = (double)current / total;
            var filled = (int)(progress * barLength);

            var bar = new string('█', filled) + new string('░', barLength - filled);

            Console.Write($"\r[{bar}] {progress * 100:0.0}% ({current}/{total})");
        }

        private static string GetTrainingResults(Network.Network net)
        {
            var stringBuilder = new StringBuilder();
            foreach (var input in Inputs)
            {
                var output = net.Forward(input);
                stringBuilder.AppendLine($"{input[0]} {input[1]}:{output[0]:F6}");
            }

            return stringBuilder.ToString();
        }
    }
}