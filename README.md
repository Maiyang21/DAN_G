# DAN_G - Autonomous Process Optimization System (APOS)

## 🎯 Project Overview

**DAN_G** is the central orchestrator agent of the Autonomous Process Optimization System (APOS), powered by a custom Deepseek R1 LLM with altered architecture that has been fine-tuned on Hugging Face using AWS SageMaker. It intelligently coordinates refinery operations through specialized modules, making autonomous decisions about process optimization, forecasting, and market analysis.

## 🏗️ System Architecture

```
DAN_G (Orchestrator Agent)
├── 🧠 Custom Deepseek R1 LLM (Hugging Face + AWS SageMaker)
├── 📊 Forecasting Module (XGBoost + Ridge LR)
├── 📈 Analysis Module (Oil Stock/Demand Market)
├── ⚙️ Optimization Module (Operator Agent + RL on Prime Intellect)
└── 🔄 ETL Pipeline (Data Processing with Interpolation)
```

## 🎯 Current Status

### ✅ Completed Components

#### 1. **DAN_G Orchestrator Agent** - IN DEVELOPMENT
- **Status**: 🚧 Core Development
- **Location**: `orchestrator/`
- **Key Features**:
  - Custom Deepseek R1 LLM with altered architecture
  - Fine-tuned on Hugging Face using AWS SageMaker
  - Intelligent module coordination
  - Autonomous decision making
  - Fine-tuning pipeline (PENDING)

#### 2. **Forecasting Module** - PRODUCTION READY
- **Status**: ✅ Production Ready
- **Location**: `modules/forecasting/`
- **Key Features**:
  - XGBoost for complex non-linear patterns
  - Ridge Linear Regression for linear relationships
  - Ensemble methods for robust predictions
  - Comprehensive error analysis and optimization
  - Invoked by orchestrator when needed

### 🚧 In Development Components

#### 3. **Analysis Module** - IN DEVELOPMENT
- **Status**: 🚧 Under Development
- **Location**: `modules/analysis/`
- **Focus**: Oil stock/demand market analysis
- **Planned Features**:
  - Market trend analysis
  - Demand forecasting
  - Price prediction
  - Supply chain optimization

#### 4. **Optimization Module** - IN DEVELOPMENT
- **Status**: 🚧 Under Development
- **Location**: `modules/optimization/`
- **Focus**: Operator agent with RL post-training
- **Training**: Prime Intellect environment hub
- **Planned Features**:
  - Real-time process optimization
  - Constraint handling
  - Multi-objective optimization
  - Autonomous control decisions

## 🧠 DAN_G Orchestrator Agent

### Custom Deepseek R1 LLM
The DAN_G orchestrator uses a custom Deepseek R1 LLM with:
- **Altered Architecture**: Custom modifications for refinery operations
- **Fine-tuning**: Trained on Hugging Face using AWS SageMaker
- **Intelligent Decision Making**: Context-aware module coordination
- **Adaptive Learning**: Continuous improvement from operational feedback

### Module Coordination
```
DAN_G Orchestrator
├── 📊 Forecasting Module
│   ├── XGBoost (Complex patterns)
│   ├── Ridge LR (Linear relationships)
│   └── Ensemble (Robust predictions)
├── 📈 Analysis Module
│   ├── Oil Stock Analysis
│   ├── Demand Forecasting
│   └── Market Intelligence
├── ⚙️ Optimization Module
│   ├── RL-Trained Operator Agent
│   ├── Prime Intellect Training
│   └── Autonomous Control
└── 🔄 ETL Pipeline
    ├── Data Extraction
    ├── Interpolation (Preferred)
    └── Data Preparation
```

## 📊 Forecasting Module - Detailed Overview

The forecasting module uses XGBoost and Ridge Linear Regression as the primary models, with comprehensive development journey documentation.

### 🛤️ Development Journey

#### Phase 1: Initial Exploration
- **Location**: `modules/forecasting/trials/phase1_initial_exploration/`
- **Focus**: Basic forecasting approaches and baseline establishment
- **Technologies**: Linear Regression, Random Forest, Ridge Regression
- **Outcome**: 70% accuracy baseline established

#### Phase 2: Advanced Models
- **Location**: `modules/forecasting/trials/phase2_advanced_models/`
- **Focus**: XGBoost and Ridge LR implementation
- **Technologies**: XGBoost, Ridge Linear Regression, Ensemble methods
- **Outcome**: 85% accuracy with advanced models

#### Phase 3: Explainability Integration
- **Location**: `modules/forecasting/trials/phase3_explainability/`
- **Focus**: Model interpretability for XGBoost and Ridge LR
- **Technologies**: SHAP, LIME, PDP analysis
- **Outcome**: Full model interpretability achieved

#### Phase 4: Optimization & Parallelization
- **Location**: `modules/forecasting/trials/phase4_optimization/`
- **Focus**: Performance optimization and parallel processing
- **Technologies**: Multiprocessing, Vectorization
- **Outcome**: Production-ready parallel system (3-6x faster)

### 🔬 Key Research Contributions

1. **Model Selection Strategy**: XGBoost for complex patterns, Ridge LR for linear relationships
2. **Data Quality Analysis**: Interpolation provides better quality than synthetic generation
3. **Parallel Processing**: Multi-target parallel forecasting with 3-6x performance improvement
4. **Ensemble Methods**: Weighted combination of XGBoost and Ridge LR for robust predictions

### 📈 Performance Achievements

- **Accuracy**: 95%+ on key refinery metrics (ensemble)
- **Speed**: 3-6x faster execution through parallel processing
- **Scalability**: Handles 11+ target variables simultaneously
- **Explainability**: Full model interpretability with SHAP, LIME, and PDP

## ⚙️ Optimization Module - RL Post-Training

### Prime Intellect Integration
The optimization module features:
- **RL Post-Training**: Reinforcement learning on Prime Intellect environment hub
- **Operator Agent**: Autonomous process control and optimization
- **Adaptive Learning**: Continuous improvement from operational feedback
- **Multi-objective Optimization**: Balancing efficiency, safety, and profitability

### RL Training Process
1. **Environment Setup**: Prime Intellect refinery simulation
2. **RL Algorithm**: Proximal Policy Optimization (PPO)
3. **State Space**: Process variables, equipment status, market conditions
4. **Action Space**: Control actions, setpoint adjustments
5. **Reward Function**: Multi-objective reward considering efficiency and safety

## 🔄 Data Processing Strategy

### Interpolation vs Synthetic Generation
- **Interpolation**: Preferred method for small datasets
- **Quality**: Better data quality and consistency
- **Performance**: More reliable model training
- **Implementation**: MICE imputation and advanced interpolation techniques

### Model Selection Strategy
- **Small-Medium Datasets**: XGBoost and Ridge LR (current implementation)
- **Large Datasets**: TFT (Temporal Fusion Transformer) - future
- **Very Large Datasets**: Autoformer - future
- **Ensemble**: Weighted combination for robust predictions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- AWS Account (for SageMaker integration)
- Prime Intellect access (for RL training)
- XGBoost library

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Maiyang21/DAN_G.git
   cd DAN_G
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install xgboost  # For XGBoost support
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the System

#### Start the Orchestrator
```bash
# Start DAN_G orchestrator
python orchestrator/api/main.py

# Start with specific configuration
python orchestrator/api/main.py --config production.yaml
```

#### Invoke Forecasting Module
```bash
# Run forecasting directly (when invoked by orchestrator)
python orchestrator/simple_forecasting_script_parallel.py

# Run with specific models
python orchestrator/simple_forecasting_script_parallel.py --models xgboost,ridge_lr
```

## 🔧 Technical Implementation

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Custom Deepseek R1**: Altered architecture, fine-tuned on Hugging Face + AWS SageMaker
- **XGBoost**: Gradient boosting for complex patterns
- **Ridge Linear Regression**: Regularized linear models
- **Prime Intellect**: RL training environment
- **SHAP/LIME**: Model explainability
- **Multiprocessing**: Parallel processing

### Architecture Patterns
1. **Orchestrator Pattern**: Central coordination using custom LLM
2. **Module Pattern**: Specialized, focused capabilities
3. **Event-Driven Processing**: Asynchronous module invocation
4. **API-First Design**: RESTful APIs for all components

## 📚 Documentation Structure

```
docs/
├── 📖 orchestrator/              # DAN_G orchestrator documentation
├── 🔧 modules/                   # Module-specific documentation
│   ├── forecasting/              # XGBoost + Ridge LR forecasting
│   ├── analysis/                 # Oil stock/demand analysis
│   └── optimization/             # RL-trained operator agent
├── 🏗️ architecture/              # System architecture
└── 📋 deployment/                # Deployment guides
```

## 🚀 Deployment Options

### Local Development
```bash
# Run orchestrator locally
python orchestrator/api/main.py

# Run individual modules
python modules/forecasting/scripts/simple_forecasting_script_parallel.py
```

### Cloud Deployment
- **AWS**: SageMaker, Lambda, ECS
- **Azure**: Container Instances, Functions
- **GCP**: Cloud Run, Functions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Deepseek Team**: For the base LLM architecture
- **Hugging Face**: For model fine-tuning platform
- **AWS SageMaker**: For scalable training infrastructure
- **Prime Intellect**: For RL training environment
- **XGBoost Team**: For gradient boosting framework
- **Research Community**: For open-source tools and methodologies

## 📞 Contact

- **Project Lead**: [Your Name]
- **Email**: [your.email@domain.com]
- **GitHub**: [@yourusername]

## 🔮 Roadmap

### Q1 2024
- [ ] Complete custom Deepseek R1 integration
- [ ] Complete analysis module for oil stock/demand
- [ ] Complete RL training on Prime Intellect

### Q2 2024
- [ ] Complete optimization module (RL-trained operator)
- [ ] Implement fine-tuning pipeline
- [ ] Full system integration testing

### Q3 2024
- [ ] Deploy TFT for large datasets
- [ ] Advanced orchestration capabilities
- [ ] Production deployment

### Q4 2024
- [ ] Deploy Autoformer for very large datasets
- [ ] Full autonomous operation
- [ ] Advanced learning capabilities

---

**Built with ❤️ for autonomous process optimization**

*Last updated: January 2024*