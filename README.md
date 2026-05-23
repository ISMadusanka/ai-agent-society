# 🏛️ AI Agent Society — Multi-Agent RL Simulation

A Python simulation where **autonomous LLM-powered agents** form and evolve their own society. Each agent has a unique personality, persistent FAISS-backed memory, and makes decisions using reinforcement learning signals. The society's structure — roles, rules, alliances, governance — emerges entirely from agent interactions.

## Key Features

- **Autonomous Agents**: Each agent has an LLM-generated personality (Big Five traits, values, backstory, speaking style)
- **FAISS Vector Memory**: Per-agent vector store with **reward-weighted retrieval** — the core RL mechanism
- **Self-Organizing Society**: Agents propose roles, create rules, form alliances, and vote — all emergently
- **Multi-Agent RL**: Actions yield rewards (engagement, influence, consistency, harmony) that shape future decisions via memory retrieval
- **Rich Dashboard**: Live terminal UI showing agent status, actions, and society evolution
- **Persistence**: Full state serialization — resume your society across sessions

## Architecture

```
ai-agent-society/
├── main.py                    # CLI entry point
├── config/
│   ├── settings.py            # Pydantic-style config management
│   └── default.yaml           # Default configuration
├── llm/
│   └── client.py              # Ollama HTTP abstraction (.call() → dict)
├── memory/
│   ├── embeddings.py          # Sentence Transformer client
│   └── vector_store.py        # FAISS index + reward-weighted retrieval
├── agents/
│   ├── agent.py               # Core Agent class (perceive→decide→learn)
│   ├── personality.py         # LLM-generated personality profiles
│   └── reflection.py          # Experience synthesis engine
├── rl/
│   ├── rewards.py             # Multi-dimensional reward functions
│   └── policy.py              # LLM policy with memory-augmented prompts
├── communication/
│   └── message.py             # Message bus (broadcast + direct)
├── society/
│   ├── relationships.py       # Directed relationship graph
│   └── governance.py          # Roles, rules, proposals, voting
└── simulation/
    └── engine.py              # Main simulation loop + Rich dashboard
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with your LLM model

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running with your model
ollama serve
ollama pull gpt-oss:20b
```

### Run

```bash
# Default: 10 agents, 100 steps
python main.py

# Customize
python main.py --agents 5 --steps 50

# Use a different model
python main.py --model "llama3:8b"

# Resume a previous society
python main.py --resume --steps 50

# Debug logging
python main.py --agents 3 --steps 10 --debug
```

## Configuration

Edit `config/default.yaml` or pass CLI flags:

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `num_agents` | `--agents` | 10 | Number of agents in the society |
| `max_steps` | `--steps` | 100 | Simulation steps to run |
| `model` | `--model` | gpt-oss:20b | Ollama model name |
| `ollama_url` | `--ollama-url` | localhost:11434 | Ollama API endpoint |
| `embed_model` | — | all-MiniLM-L6-v2 | Sentence transformer model |
| `reflection_interval` | — | 5 | Steps between reflection rounds |
| `persist_interval` | — | 10 | Steps between state saves |

## How It Works

### The RL Loop

1. **Perceive**: Agent receives broadcasts and direct messages → stored in FAISS
2. **Retrieve**: Reward-weighted memory retrieval surfaces high-value past experiences
3. **Decide**: LLM generates an action given personality + memories + society state
4. **Act**: Action is executed (speak, propose, vote, ally, challenge, etc.)
5. **Reward**: Multi-dimensional reward computed (engagement, influence, consistency, harmony)
6. **Learn**: Action + reward stored in FAISS → influences future retrievals

### Action Space

| Action | Description |
|--------|-------------|
| `speak` | Broadcast a message to all agents |
| `direct_message` | Private message to a specific agent |
| `propose_role` | Propose a role (leader, mediator, etc.) for self or another |
| `propose_rule` | Suggest a new societal rule or norm |
| `vote` | Vote for or against a pending proposal |
| `form_alliance` | Propose an alliance with another agent |
| `reflect` | Internal reflection on recent experiences |
| `observe` | Quietly observe without acting |
| `challenge` | Challenge an existing rule or role |

### Reward Components

- **Engagement** (25%): Other agents respond to your messages
- **Influence** (20%): Your proposals get accepted
- **Consistency** (20%): Acting in line with your personality traits
- **Harmony** (20%): Society develops structure (roles, rules)
- **Alliance** (15%): Maintaining positive relationships

## License

MIT
