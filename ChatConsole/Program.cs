using System.Collections.Concurrent;
using System.Reflection;
using System.Text.Json;

using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.ML.Tokenizers;

// --- Configuration ---
// Default model, can be changed by the user.
// string modelName = "llama3.2:latest";
string modelName = "gemma3:12b";
Uri endpoint = new("http://localhost:11434");
// --- End Configuration ---

// --- Command Line Arguments ---
if (args.Length > 0)
{
    for (int i = 0; i < args.Length; i++)
    {
        if (args[i] == "--endpoint" && i + 1 < args.Length)
        {
            if (Uri.TryCreate(args[i + 1], UriKind.Absolute, out Uri? uri))
            {
                endpoint = uri;

                Console.WriteLine($"Using endpoint from command line: {endpoint}");

                i++; // Skip the next argument since it's the value for --endpoint
            }
            else
            {
                Console.WriteLine($"Invalid endpoint URI: {args[i + 1]}. Using default endpoint.");
            }
        }
    }
}
// --- End Command Line Arguments ---

// --- Constants ---
const string SummaryPrompt = """
    You are an AI agent dedicated to summarizing conversation histories provided to you. You generate detailed summaries that capture the full context of the
    conversation. You respond in plain text.
    """;

const string ExitCommand = "bye";
const string HelpCommand = "help";
// --- End Constants ---
string SystemPrompt = "";
string AgentPrompt = "";

bool AutonomousMode = false;
bool StepMode = true;

IConfiguration config = new ConfigurationBuilder()
    .SetBasePath(AppContext.BaseDirectory)
    .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
    .Build();

if(config["SystemPrompt"] is string systemPrompt)
    SystemPrompt = systemPrompt;

if(config["AgentPrompt"] is string agentPrompt)
    AgentPrompt = agentPrompt;

if(bool.TryParse(config["AutonomousMode"], out var autonomousMode))
    AutonomousMode = autonomousMode;

if(bool.TryParse(config["StepMode"], out var stepMode))
    StepMode = stepMode;

// Ensure Ollama is running and you have a model pulled, e.g., "ollama pull mistral-small3.1:latest"
ConcurrentDictionary<string, IChatClient> builtClients = new();

var defaultClientBuilder = new ChatClientBuilder(new OllamaChatClient(endpoint, modelName));
var client = builtClients.GetOrAdd(modelName, defaultClientBuilder.Build());

JsonSerializerOptions prettyJsonOptions = new() { WriteIndented = true };

Console.WriteLine($"Starting chat with Ollama compatible model: {modelName}");
Console.WriteLine($"Type '/{ExitCommand}' to end the conversation.");
Console.WriteLine("Ensure Ollama is running and the specified model is available wherever the operator is hosting it.");
Console.WriteLine($"Type '/{HelpCommand}' for a list of available commands.");
Console.WriteLine("----------------------------------------------------");

// Maintain a list of messages for context
List<ChatMessage> conversationHistory = [
    new ChatMessage(ChatRole.System, SystemPrompt)
];

List<ChatMessage> autoHistory = [
    new ChatMessage(ChatRole.System, AgentPrompt)
];

var LoadTiktokenTokenizerFromResourceAsync = async (string resourceName, CancellationToken cancellationToken = default) =>
{
    var assembly = Assembly.GetExecutingAssembly();

    using var resourceStream = assembly.GetManifestResourceStream(resourceName);

    if (resourceStream is null)
    {
        Console.WriteLine($"Warning: Embedded resource: {resourceName} not found. Token estimates will be unavailable for models that depend on it.");

        return null;
    }

    return await TiktokenTokenizer.CreateAsync(resourceStream, null, null, cancellationToken: cancellationToken);
};

CancellationTokenSource cts = new();

// serialize conversationHistory to JSON and save to specified file name/path, if not specified, save to local directory with generated name
Func<string?, CancellationToken, Task> saveCommand = async (fileName, cancellationToken) =>
{
    var fileNameToSave = fileName ?? $"chat_{DateTime.Now:yyyyMMdd_HHmmss}.json";
    var filePath = fileName is null ? Path.Combine(Directory.GetCurrentDirectory(), fileNameToSave) : Path.GetFullPath(fileNameToSave);
    var json = JsonSerializer.Serialize(conversationHistory, prettyJsonOptions);

    await File.WriteAllTextAsync(filePath, json, cancellationToken);

    Console.WriteLine($"Conversation saved to {filePath}");
};

Func<string?, CancellationToken, Task> loadCommand = async (fileName, cancellationToken) =>
{
    if (string.IsNullOrWhiteSpace(fileName))
    {
        Console.WriteLine("File name is required for loading a conversation.");

        return;
    }

    try
    {
        var filePath = Path.GetFullPath(fileName);
        var json = await File.ReadAllTextAsync(filePath, cancellationToken);
        var loadedHistory = JsonSerializer.Deserialize<List<ChatMessage>>(json);

        if (loadedHistory is null)
        {
            Console.WriteLine($"Failed to load conversation from {filePath}. Invalid file format.");

            return;
        }

        conversationHistory = loadedHistory;

        Console.WriteLine($"Conversation loaded from {filePath}");
        Console.WriteLine($"Last message ({conversationHistory[^1].Role}):");

        foreach (var item in conversationHistory.Last().Contents)
        {
            Console.WriteLine(item);
        }
    }
    catch (FileNotFoundException)
    {
        Console.WriteLine($"File not found: {fileName}");
    }
    catch (JsonException ex)
    {
        Console.WriteLine($"Error deserializing file: {ex.Message}");
    }
};

Func<string?, CancellationToken, Task> forgetCommand = (_, _) =>
{
    conversationHistory = [new ChatMessage(ChatRole.System, SystemPrompt)];

    Console.WriteLine("Conversation history cleared.");

    return Task.CompletedTask;
};

var estimator = await LoadTiktokenTokenizerFromResourceAsync("ChatConsole.Resources.llama-3.2-3b.tokenizer.model", cts.Token);

// Interesting stats:
// Conversation length (in terms of bytes of serialized json)
// Number of turns (count instances of ChatRole.User in conversationHistory)
// Estimated input token count
Func<string?, CancellationToken, Task> statsCommand = async (_, cancellationToken) =>
{
    await using var memStream = new MemoryStream();

    await JsonSerializer.SerializeAsync(memStream, conversationHistory, cancellationToken: cancellationToken);

    var byteCount = memStream.Length;
    var turnCount = conversationHistory.Count(m => m.Role == ChatRole.User);

    memStream.Position = 0;

    using var reader = new StreamReader(memStream);

    var json = await reader.ReadToEndAsync(cancellationToken);
    var estimatedTokenCount = estimator.CountTokens(json);

    var stats = new
    {
        ByteCount = byteCount,
        TurnCount = turnCount,
        EstimatedTokenCount = estimatedTokenCount
    };

    memStream.SetLength(0);

    await JsonSerializer.SerializeAsync(memStream, stats, prettyJsonOptions, cancellationToken);

    memStream.Position = 0;

    using var prettyReader = new StreamReader(memStream);

    var pretty = await prettyReader.ReadToEndAsync(cancellationToken);

    Console.WriteLine(pretty);
};

Func<string?, CancellationToken, Task> changeModelCommand = (newModel, _) =>
{
    if (string.IsNullOrWhiteSpace(newModel))
    {
        Console.WriteLine("New model name is required.");

        return Task.CompletedTask;
    }

    modelName = newModel;

    client = builtClients.GetOrAdd(modelName, new ChatClientBuilder(new OllamaChatClient(endpoint, modelName)).Build());

    Console.WriteLine($"Model changed to: {modelName}");

    return Task.CompletedTask;
};

Func<string?, CancellationToken, Task> summarizeCommand = async (_, cancellationToken) =>
{
    List<ChatMessage> summaryHistory = [new ChatMessage(ChatRole.System, SummaryPrompt)];

    await using MemoryStream memStream = new();

    await JsonSerializer.SerializeAsync(memStream, conversationHistory, cancellationToken: cancellationToken);

    memStream.Position = 0;

    using StreamReader reader = new(memStream);

    summaryHistory.Add(new(ChatRole.User, await reader.ReadToEndAsync(cancellationToken)));

    try
    {
        var response = await client.GetResponseAsync(summaryHistory, cancellationToken: cancellationToken);

        if (response is null)
        {
            Console.WriteLine("Failed to generate summary.");
            return;
        }

        var newPrompt = string.Concat(SystemPrompt, "\n\nSummary of the conversation so far:\n", response.Text);

        conversationHistory = [new ChatMessage(ChatRole.System, newPrompt)];

        Console.WriteLine(response.Text);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: Could not connect to Ollama or process the request. Details: {ex.Message}");
        Console.WriteLine("Please ensure Ollama is running and the model name is correct.");
    }
    Console.WriteLine("----------------------------------------------------");
};

Func<string?, CancellationToken, Task> replaceSystemPromptCommand = (newPrompt, _) =>
{
    if (string.IsNullOrWhiteSpace(newPrompt))
    {
        Console.WriteLine("New system prompt is required.");
        return Task.CompletedTask;
    }

    if (conversationHistory.Count > 0 && conversationHistory[0].Role == ChatRole.System)
    {
        conversationHistory[0] = new ChatMessage(ChatRole.System, newPrompt);
    }
    else
    {
        conversationHistory.Insert(0, new ChatMessage(ChatRole.System, newPrompt));
    }

    Console.WriteLine("System prompt replaced.");

    return Task.CompletedTask;
};


Func<string?, CancellationToken, Task> exitCommand = (_, cancellationToken) =>
{
    Console.WriteLine("Exiting chat.");

    cts.Cancel();

    return Task.CompletedTask;
};

Func<string?, CancellationToken, Task> helpCommand = (_, _) =>
{
    Console.WriteLine("Available commands:");
    Console.WriteLine("\t/save [filename] - Save the current conversation to a file. If no filename is provided, a default name will be used.");
    Console.WriteLine("\t/load <filename> - Load a conversation from a file.");
    Console.WriteLine("\t/stats - Display conversation statistics.");
    Console.WriteLine("\t/model <modelname> - Change the current model.");
    Console.WriteLine("\t/forget - Clear the conversation history.");
    Console.WriteLine("\t/summarize - Summarize and truncate the conversation history.");
    Console.WriteLine("\t/system <prompt> - Replace the system prompt.");
    Console.WriteLine($"\t/{ExitCommand} - Exit the application.");
    Console.WriteLine("\t/help - Show this help message.");

    return Task.CompletedTask;
};

Func<string?, CancellationToken, Task> defaultCommand = (_, _) =>
{
    Console.WriteLine($"Unknown command. Type '/{ExitCommand}' to end the conversation.");

    return Task.CompletedTask;
};

Func<string?, CancellationToken, Task> autonomousCommand = (_, _) =>
{
    Console.WriteLine("Activating autonomous mode.");
    AutonomousMode = true;

    return Task.CompletedTask;
};

Func<string, Func<string?, CancellationToken, Task>> getCommand = command =>
{
    return command switch
    {
        "save" => saveCommand,
        "load" => loadCommand,
        "stats" => statsCommand,
        "model" => changeModelCommand,
        "forget" => forgetCommand,
        "summarize" => summarizeCommand,
        "help" => helpCommand,
        "system" => replaceSystemPromptCommand,
        "auto" => autonomousCommand,
        ExitCommand => exitCommand,
        _ => defaultCommand
    };
};

var parseCommand = (string input) =>
{
    var parts = input.Split(' ', 2);
    var command = parts[0][1..];
    var argument = parts.Length > 1 ? parts[1] : null;

    return (command, argument);
};

Console.CancelKeyPress += (sender, e) =>
{
    Console.WriteLine("Cancellation requested. Exiting...");

    cts.Cancel();
    e.Cancel = true;
};

Func<CancellationToken, Task<bool>> doRound = async (cancellationToken) =>
{
    Console.WriteLine("You:");

    string? userInput = Console.ReadLine();

    if (string.IsNullOrWhiteSpace(userInput))
    {
        return true;
    }

    if (userInput.StartsWith("/", StringComparison.OrdinalIgnoreCase))
    {
        var (command, argument) = parseCommand(userInput);

        var handler = getCommand(command);

        await handler(argument, cancellationToken);

        return true;
    }

    // Add user's message to history
    conversationHistory.Add(new ChatMessage(ChatRole.User, userInput));

    try
    {
        Console.WriteLine($"{modelName}:");

        var response = string.Empty;

        await foreach (ChatResponseUpdate item in client.GetStreamingResponseAsync(conversationHistory, cancellationToken: cancellationToken))
        {
            Console.Write(item.Text);
            response += item.Text;
        }

        conversationHistory.Add(new ChatMessage(ChatRole.Assistant, response));

        Console.WriteLine();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: Could not connect to Ollama or process the request. Details: {ex.Message}");
        Console.WriteLine("Please ensure Ollama is running and the model name is correct.");
        Console.WriteLine("Call the operator and complain that he hasn't pulled a model you need.");

        // Remove the last user message from history if the call failed
        if (conversationHistory.Count > 0 && conversationHistory[^1].Role == ChatRole.User)
        {
            conversationHistory.RemoveAt(conversationHistory.Count - 1);
            Console.WriteLine("Your last message was not sent due to the error. Please try again.");
        }

        return true;
    }

    Console.WriteLine("----------------------------------------------------");

    return true;
};

Func<string?, CancellationToken, Task<bool>> doAutoRound = async (direction, cancellationToken) =>
{
    if (conversationHistory.Count == 1)
    {
        var initialMessage = direction ?? "Hello.";

        conversationHistory.Add(new ChatMessage(ChatRole.User, initialMessage));
        autoHistory.Add(new ChatMessage(ChatRole.Assistant, initialMessage));

        Console.WriteLine($"Agent:\n{initialMessage}\n");

        direction = null;
    }

    Console.WriteLine($"{modelName}:");

    var response = direction is null ? string.Empty : "direction: " + direction;

    try
    {
        if (direction is null)
        {
            await foreach (ChatResponseUpdate item in client.GetStreamingResponseAsync(conversationHistory, cancellationToken: cancellationToken))
            {
                Console.Write(item.Text);
                response += item.Text;
            }

            conversationHistory.Add(new ChatMessage(ChatRole.Assistant, response));
        }

        autoHistory.Add(new ChatMessage(ChatRole.User, response));

        Console.WriteLine($"\n\nAgent ({modelName}):");

        response = string.Empty;

        await foreach (ChatResponseUpdate item in client.GetStreamingResponseAsync(autoHistory, cancellationToken: cancellationToken))
        {
            Console.Write(item.Text);
            response += item.Text;
        }

        autoHistory.Add(new ChatMessage(ChatRole.Assistant, response));
        conversationHistory.Add(new ChatMessage(ChatRole.User, response));

        Console.WriteLine();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: Could not connect to Ollama or process the request. Details: {ex.Message}");
        Console.WriteLine("Please ensure Ollama is running and the model name is correct.");
        Console.WriteLine("Call the operator and complain that he hasn't pulled a model you need.");

        // Remove the last user message from history if the call failed
        if (conversationHistory.Count > 0 && conversationHistory[^1].Role == ChatRole.User)
        {
            conversationHistory.RemoveAt(conversationHistory.Count - 1);
            Console.WriteLine("Your last message was not sent due to the error. Please try again.");
        }

        return true;
    }

    Console.WriteLine("----------------------------------------------------");

    return true;
};

while (!cts.IsCancellationRequested)
{
    string? direction = null;

    if (AutonomousMode && StepMode)
    {
        Console.WriteLine("Type '/auto' to exit autonomous mode");
        Console.WriteLine("Type '/direct' followed by a prompt to direct the agent");

        var userInput = Console.ReadLine();

        if (!string.IsNullOrWhiteSpace(userInput))
        {
            if (userInput.StartsWith("/auto", StringComparison.OrdinalIgnoreCase))
            {
                AutonomousMode = false;

                Console.WriteLine("Exiting autonomous mode.");

                continue;
            }

            if (userInput.StartsWith("/direct", StringComparison.OrdinalIgnoreCase))
            {
                direction = userInput[7..].TrimStart();
            }
        }
    }

    var result = AutonomousMode ? doAutoRound(direction, cts.Token) : doRound(cts.Token);

    if (await result)
        continue;

    break;
}