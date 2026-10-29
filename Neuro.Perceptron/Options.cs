using CommandLine;

namespace Neuro.Perceptron;

public class Options
{
    [Option('e', "epochs", Required = true, HelpText = "Epoches")]
    public int EpochCount { get; set; }

    [Option('r', "learningRate", Required = true, HelpText = "Learning rate")]
    public double LearningRate { get; set; }

    [Option('f', "writeToFile", Required = false, HelpText = "Write to file")]
    public bool WriteToFile { get; set; }
    
    [Option('l', "layers", Separator = ',', Required = true, HelpText = "Layers")]
    public IEnumerable<int> Layers { get; set; } = Enumerable.Empty<int>();
}