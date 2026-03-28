namespace Neuro.Functions;

public interface IFunction
{
    public abstract double Value(double x);
    public abstract double Gradient(double x);
}