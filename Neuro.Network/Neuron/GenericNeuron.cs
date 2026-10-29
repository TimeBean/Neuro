using Neuro.Functions;

namespace Neuro.Network.Neuron;

public class GenericNeuron : Neuron
{
    private readonly IFunction _activationFunction;
    private double _lastWeightedSum;

    public GenericNeuron(int inputCount, IFunction activationFunction)
        : base(inputCount)
    {
        _activationFunction = activationFunction;
    }

    public override double Forward(double[] inputs)
    {
        _lastWeightedSum = Bias;
        for (var i = 0; i < inputs.Length; i++)
        {
            _lastWeightedSum += inputs[i] * Weights[i];
        }

        return _activationFunction.Value(_lastWeightedSum);
    }

    public override void UpdateWeights(double[] inputs, double delta, double learningRate)
    {
        for (var i = 0; i < Weights.Length; i++)
        {
            Weights[i] += learningRate * delta * inputs[i];
        }

        Bias += learningRate * delta;
    }

    public double GetLastWeightedSum() => _lastWeightedSum;
}