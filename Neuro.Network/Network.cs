using System;
using Neuro.Functions;
using Neuro.Network.Neuron;

namespace Neuro.Network
{
    public class Network
    {
        private readonly Layer[] _layers;
        private readonly double _learningRate;
        private readonly IFunction _activationFunction;

        public Network(double learningRate, IFunction activationFunction, params int[] layerSizes)
        {
            _learningRate = learningRate;
            _activationFunction = activationFunction;

            _layers = new Layer[layerSizes.Length - 1];
            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i] = new Layer(layerSizes[i + 1], layerSizes[i], _activationFunction);
            }
        }

        public double[] Forward(double[] inputs)
        {
            double[] output = inputs;
            foreach (var layer in _layers)
                output = layer.Forward(output);
            return output;
        }

        public void Train(double[] inputs, double[] targets)
        {
            var output = Forward(inputs);

            var deltas = new double[_layers.Length][];

            int last = _layers.Length - 1;
            deltas[last] = new double[_layers[last].Neurons.Length];

            for (int i = 0; i < deltas[last].Length; i++)
            {
                var neuron = (GenericNeuron)_layers[last].Neurons[i];
                double weightedSum = neuron.GetLastWeightedSum();
                deltas[last][i] = (targets[i] - output[i]) * _activationFunction.Gradient(weightedSum);
            }

            for (int l = last - 1; l >= 0; l--)
            {
                deltas[l] = new double[_layers[l].Neurons.Length];

                for (int i = 0; i < _layers[l].Neurons.Length; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < _layers[l + 1].Neurons.Length; j++)
                    {
                        sum += deltas[l + 1][j] * _layers[l + 1].Neurons[j].Weights[i];
                    }

                    var neuron = (GenericNeuron)_layers[l].Neurons[i];
                    double weightedSum = neuron.GetLastWeightedSum();
                    deltas[l][i] = sum * _activationFunction.Gradient(weightedSum);
                }
            }

            var layerInput = inputs;
            for (int l = 0; l < _layers.Length; l++)
            {
                for (int n = 0; n < _layers[l].Neurons.Length; n++)
                {
                    _layers[l].Neurons[n].UpdateWeights(layerInput, deltas[l][n], _learningRate);
                }
                layerInput = _layers[l].Outputs;
            }
        }
    }
}