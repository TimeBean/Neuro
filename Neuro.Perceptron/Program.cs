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
            [1.0],
            [1.0],
            [0.0]
        ];

        private static void Main()
        {
            IFunction sigmoid = new Sigmoid();

            var net = new Network.Network(learningRate: 0.1, activationFunction: sigmoid, 2, 16, 1);

            for (var epoch = 0; epoch < 50000; epoch++)
            {
                for (var i = 0; i < Inputs.Length; i++)
                    net.Train(Inputs[i], Targets[i]);
            }

            Console.WriteLine("Results after training:");
            foreach (var input in Inputs)
            {
                var output = net.Forward(input);
                Console.WriteLine($"Input: {input[0]}, {input[1]} -> Output: {output[0]:F6}");
            }
        }
    }
}