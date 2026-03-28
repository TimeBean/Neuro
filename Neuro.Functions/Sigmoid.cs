namespace Neuro.Functions;

public class Sigmoid : IFunction
{
    public double Value(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    public double Gradient(double x)
    {
        var y = Value(x);
        return y * (1 - y);
    }
}