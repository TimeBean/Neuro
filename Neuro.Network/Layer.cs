using Neuro.Functions;

namespace Neuro.Network;

public class Layer
{
    public Neuron[] Neurons;
    public double[] Outputs { get; private set; }
    
    private readonly IFunction _activationFunction;

    public Layer(int neuronCount, int inputsPerNeuron, IFunction activationFunction)
    {
        _activationFunction = activationFunction;

        Neurons = new Neuron[neuronCount];
        Outputs = new double[neuronCount];

        for (var i = 0; i < neuronCount; i++)
        {
            Neurons[i] = new GenericNeuron(inputsPerNeuron, _activationFunction);
        }
    }

    public double[] Forward(double[] inputs)
    {
        for (var i = 0; i < Neurons.Length; i++)
        {
            Outputs[i] = Neurons[i].Forward(inputs);
        }

        return Outputs;
    }
}