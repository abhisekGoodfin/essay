# CFA Essay Evaluation System

The CFA Essay Evaluation System is a Python-based application designed to evaluate and compare responses to CFA Level 3 essay questions using various large language models (LLMs). It supports multiple LLM providers, calculates evaluation metrics, and generates visualizations for model performance comparison.

## Features

- **LLM Integration**: Supports multiple LLM providers, including OpenAI, Anthropic, AWS Bedrock, Groq, Writer, Gemini, and XAI.
- **Prompt Templates**: Includes customizable prompt templates for generating answers and grading responses.
- **Metrics Calculation**: Computes metrics such as cosine similarity, cost estimation, and grading success rates.
- **Visualization**: Generates scatter plots, radar charts, and bar charts for model performance comparison.
- **Cost Estimation**: Estimates API costs based on token usage and predefined pricing.
- **Configurable**: Easily configurable through the `config.py` file.

## Project Structure

```
cfa_essay_evaluation/
│
├── config.py               # Configuration file for API keys and settings
├── main.py                 # Main application file
├── requirements.txt         # Python package dependencies
│
├── llms/                   # Directory for LLM-related modules
│   ├── __init__.py
│   ├── openai_llm.py        # OpenAI LLM integration
│   ├── anthropic_llm.py     # Anthropic LLM integration
│   ├── aws_bedrock_llm.py   # AWS Bedrock LLM integration
│   ├── groq_llm.py          # Groq LLM integration
│   ├── writer_llm.py        # Writer LLM integration
│   ├── gemini_llm.py        # Gemini LLM integration
│   └── xai_llm.py           # XAI LLM integration
│
├── prompts/                # Directory for prompt templates
│   ├── __init__.py
│   ├── answer_prompts.py    # Answer generation prompts
│   └── grading_prompts.py   # Grading prompts
│
├── metrics/                # Directory for metrics calculation modules
│   ├── __init__.py
│   ├── cosine_similarity.py  # Cosine similarity calculation
│   ├── cost_estimation.py    # Cost estimation calculation
│   └── grading_success.py    # Grading success rate calculation
│
└── visualizations/         # Directory for visualization modules
    ├── __init__.py
    ├── scatter_plot.py      # Scatter plot visualization
    ├── radar_chart.py       # Radar chart visualization
    └── bar_chart.py         # Bar chart visualization
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/cfa_essay_evaluation.git
   cd cfa_essay_evaluation
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure your API keys and settings in the `config.py` file.

### Setting Up a Python Virtual Environment

1. Create a virtual environment in the current directory:

   ```bash
   python3 -m venv .venv
   ```

## Usage

1. Run the main application:

   ```bash
   python main.py
   ```

2. Follow the on-screen instructions to evaluate and compare CFA Level 3 essay responses.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request describing your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.