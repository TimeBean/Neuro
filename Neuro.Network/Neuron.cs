namespace Neuro.Network;

public abstract class Neuron
{
    public abstract double Forward(double[] inputs);
    public abstract void UpdateWeights(double[] inputs, double delta, double learningRate);

    protected Neuron(int inputCount)
    {
        Bias = Random.NextDouble() - 0.5;

        Weights = new double[inputCount];
        for (var i = 0; i < inputCount; i++)
        {
            Weights[i] = Random.NextDouble() - 0.5;
        }
    }

    public double[] Weights { get; set; }
    public double Bias { get; set; }

    private static readonly Random Random = new Random(52);
}