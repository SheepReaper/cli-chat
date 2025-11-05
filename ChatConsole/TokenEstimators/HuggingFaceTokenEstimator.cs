using System.Reflection;
using Microsoft.ML.Tokenizers;

namespace ChatConsole.TokenEstimators;

[CompatibleModel("llama3.2:latest")]
internal sealed class HuggingFaceTokenEstimator : ITokenEstimator
{
    private const string TokenizerResourceName = "ChatConsole.Resources.llama-3.2-3b.tokenizer.model";
    private readonly Tokenizer? _tokenizer = null;


    public HuggingFaceTokenEstimator()
    {
        var assembly = Assembly.GetExecutingAssembly();

        using var resourceStream = assembly.GetManifestResourceStream(TokenizerResourceName);

        if (resourceStream is null)
        {
            Console.WriteLine($"Warning: Embedded resource: {TokenizerResourceName} not found. Token estimates will be unavailable for models that depend on it.");

            return;
        }

        _tokenizer = TiktokenTokenizer.Create(resourceStream, null, null);
    }

    public int EstimateTokens(ReadOnlySpan<char> jsonInput)
    {
        if (_tokenizer is null)
        {
            Console.WriteLine("Warning: The token Estimator for this model is not initialized.");

            return 0;
        }

        if (jsonInput.IsEmpty || jsonInput.IsWhiteSpace()) return 0;

        return _tokenizer.CountTokens(jsonInput);
    }
}