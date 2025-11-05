namespace ChatConsole;

internal interface ITokenEstimator
{
    int EstimateTokens(ReadOnlySpan<char> jsonInput);
}

[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
internal class CompatibleModelAttribute(string modelId) : Attribute
{
    public string ModelId { get; } = modelId;
}
