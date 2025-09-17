# Project Structure - Autonomous Process Optimization System

## 📁 Repository Structure

```
DAN_G/
├── 📚 README.md                          # Main project documentation
├── 📁 agents/                            # Autonomous agents
│   ├── 📁 forecasting/                   # Forecasting Agent (COMPLETED)
│   │   ├── 📁 scripts/                   # Production scripts
│   │   ├── 📁 notebooks/                 # Jupyter notebooks
│   │   ├── 📁 trials/                    # Development phases
│   │   │   ├── 📁 phase1_initial_exploration/
│   │   │   ├── 📁 phase2_advanced_models/
│   │   │   ├── 📁 phase3_explainability/
│   │   │   └── 📁 phase4_optimization/
│   │   ├── 📁 error_analysis/            # Error analysis and visualizations
│   │   ├── 📁 models/                    # Trained models and artifacts
│   │   ├── 📁 docs/                      # Forecasting documentation
│   │   └── 📄 README.md                  # Forecasting agent documentation
│   ├── 📁 etl/                          # ETL Agent (IN DEVELOPMENT)
│   │   ├── 📁 app/                      # ETL API application
│   │   ├── 📁 data/                     # Processed data outputs
│   │   ├── 📁 scripts/                  # ETL processing scripts
│   │   └── 📄 README.md                 # ETL agent documentation
│   ├── 📁 analysis/                     # Analysis Agent (PLANNED)
│   │   └── 📄 README.md                 # Analysis agent documentation
│   ├── 📁 optimization/                 # Optimization Agent (PLANNED)
│   │   └── 📄 README.md                 # Optimization agent documentation
│   └── 📁 monitoring/                   # Monitoring Agent (PLANNED)
│       └── 📄 README.md                 # Monitoring agent documentation
├── 📁 app/                              # Main web application
│   ├── 📁 core/                         # Core utilities
│   ├── 📁 etl/                          # ETL pipeline integration
│   ├── 📄 main.py                       # Main Streamlit application
│   ├── 📄 config.py                     # Configuration settings
│   └── 📄 requirements.txt              # Python dependencies
├── 📁 docs/                             # Project documentation
│   ├── 📄 architecture.md               # System architecture
│   ├── 📄 project_structure.md          # This file
│   └── 📄 deployment.md                 # Deployment guide
├── 📁 data/                             # Data storage
│   ├── 📁 raw/                          # Raw data files
│   ├── 📁 processed/                    # Processed data
│   └── 📁 custom/                       # Custom datasets
├── 📁 models/                           # Global model storage
│   ├── 📁 trained/                      # Trained models
│   ├── 📁 artifacts/                    # Model artifacts
│   └── 📁 checkpoints/                  # Model checkpoints
├── 📁 tools/                            # Development tools
│   ├── 📁 tooling_gabs/                 # Utility tools
│   └── 📁 scripts/                      # Utility scripts
└── 📁 tests/                            # Test suites
    ├── 📁 unit/                         # Unit tests
    ├── 📁 integration/                  # Integration tests
    └── 📁 e2e/                          # End-to-end tests
```

## 🎯 Agent-Specific Structure

### Forecasting Agent (✅ COMPLETED)
```
forecasting/
├── 📄 README.md                         # Comprehensive documentation
├── 📁 scripts/                          # Production-ready scripts
│   ├── 📄 simple_forecasting_script_parallel.py  # Main parallel script
│   ├── 📄 autoformer_ebm_web_app_aws_ready.py    # AWS web app
│   ├── 📄 final_autoformer_ebm_notebook.py       # Final implementation
│   ├── 📄 performance_comparison.py              # Performance testing
│   ├── 📄 analyze_naphtha_errors.py              # Error analysis
│   └── 📄 naphtha_error_analysis.py              # Specific error analysis
├── 📁 notebooks/                        # Jupyter notebooks
│   ├── 📄 FINAL_Autoformer_EBM_Notebook.ipynb
│   └── 📄 HuggingFace_Autoformer_EBM_Notebook.ipynb
├── 📁 trials/                           # Development journey
│   ├── 📁 phase1_initial_exploration/   # Basic models and exploration
│   │   ├── 📄 README.md                 # Phase 1 documentation
│   │   └── 📄 *.ipynb                   # Initial exploration notebooks
│   ├── 📁 phase2_advanced_models/       # Deep learning and transformers
│   │   ├── 📄 README.md                 # Phase 2 documentation
│   │   └── 📄 *.ipynb                   # Advanced model notebooks
│   ├── 📁 phase3_explainability/        # Model interpretability
│   │   └── 📄 README.md                 # Phase 3 documentation
│   └── 📁 phase4_optimization/          # Performance optimization
│       └── 📄 README.md                 # Phase 4 documentation
├── 📁 error_analysis/                   # Error analysis and visualizations
│   ├── 📄 README.md                     # Error analysis documentation
│   ├── 📄 error_analysis_*.png          # Individual model error analysis
│   ├── 📄 feature_importance_*.png      # Feature importance plots
│   ├── 📄 performance_comparison.png    # Model performance comparison
│   ├── 📁 Shap_gabs/                    # SHAP analysis visualizations
│   ├── 📁 LIME_gabs/                    # LIME analysis visualizations
│   └── 📁 PDP_gabs/                     # Partial Dependence Plots
├── 📁 models/                           # Trained models and artifacts
└── 📁 docs/                             # Additional documentation
```

### ETL Agent (🚧 IN DEVELOPMENT)
```
etl/
├── 📄 README.md                         # ETL agent documentation
├── 📁 app/                              # ETL API application
│   ├── 📁 core/                         # Core utilities
│   ├── 📁 processing/                   # ETL processing modules
│   └── 📄 main_etl_pipeline.py          # FastAPI ETL service
├── 📁 data/                             # Processed data outputs
│   ├── 📁 processed/                    # Latest processed data
│   ├── 📁 processed_feb/                # February 2024 data
│   ├── 📁 processed_july/               # July 2024 data
│   └── 📁 processed_june/               # June 2024 data
├── 📁 scripts/                          # ETL processing scripts
│   ├── 📄 DANGlocal_etl_run.py          # Main ETL pipeline
│   ├── 📄 PO_CDU_ETL_PIPELINE.py        # CDU-specific ETL
│   ├── 📄 PO_CDU_TABtraction.py         # Data extraction
│   └── 📄 debug_etl.py                  # ETL debugging tools
└── 📁 docs/                             # ETL documentation
```

### Analysis Agent (📋 PLANNED)
```
analysis/
├── 📄 README.md                         # Analysis agent documentation
├── 📁 statistical/                      # Statistical analysis modules
├── 📁 pattern_detection/                # Pattern recognition
├── 📁 visualization/                    # Data visualization
├── 📁 machine_learning/                 # ML analysis
└── 📁 docs/                             # Analysis documentation
```

### Optimization Agent (📋 PLANNED)
```
optimization/
├── 📄 README.md                         # Optimization agent documentation
├── 📁 algorithms/                       # Optimization algorithms
├── 📁 constraints/                      # Constraint handling
├── 📁 objectives/                       # Objective functions
├── 📁 decision_making/                  # Autonomous decisions
└── 📁 docs/                             # Optimization documentation
```

### Monitoring Agent (📋 PLANNED)
```
monitoring/
├── 📄 README.md                         # Monitoring agent documentation
├── 📁 metrics/                          # Metrics collection
├── 📁 alerts/                           # Alerting system
├── 📁 dashboards/                       # Monitoring dashboards
├── 📁 health_checks/                    # Health monitoring
└── 📁 docs/                             # Monitoring documentation
```

## 📊 Data Organization

### Data Flow Structure
```
data/
├── 📁 raw/                              # Raw data storage
│   ├── 📁 excel/                        # Excel files
│   ├── 📁 csv/                          # CSV files
│   └── 📁 database/                     # Database exports
├── 📁 processed/                        # Processed data
│   ├── 📁 etl_output/                   # ETL processed data
│   ├── 📁 feature_engineered/           # Feature-engineered data
│   └── 📁 model_ready/                  # Model-ready datasets
├── 📁 custom/                           # Custom datasets
│   ├── 📄 custom.csv                    # Custom dataset
│   ├── 📄 custom.json                   # Custom configuration
│   └── 📄 custom.txt                    # Custom metadata
└── 📁 archived/                         # Archived data
    ├── 📁 2023/                         # 2023 data archive
    └── 📁 2024/                         # 2024 data archive
```

### Model Storage Structure
```
models/
├── 📁 trained/                          # Trained models
│   ├── 📁 forecasting/                  # Forecasting models
│   ├── 📁 etl/                          # ETL models
│   ├── 📁 analysis/                     # Analysis models
│   └── 📁 optimization/                 # Optimization models
├── 📁 artifacts/                        # Model artifacts
│   ├── 📁 scalers/                      # Data scalers
│   ├── 📁 encoders/                     # Feature encoders
│   ├── 📁 transformers/                 # Data transformers
│   └── 📁 metadata/                     # Model metadata
└── 📁 checkpoints/                      # Model checkpoints
    ├── 📁 forecasting/                  # Forecasting checkpoints
    └── 📁 etl/                          # ETL checkpoints
```

## 🛠️ Development Tools Structure

### Tooling Organization
```
tools/
├── 📁 tooling_gabs/                     # Utility tools
│   ├── 📄 *.py                          # Python utility scripts
│   ├── 📄 *.bat                         # Windows batch files
│   └── 📄 *.txt                         # Configuration files
├── 📁 scripts/                          # Utility scripts
│   ├── 📄 setup.py                      # Setup script
│   ├── 📄 deploy.py                     # Deployment script
│   └── 📄 test.py                       # Testing script
└── 📁 configs/                          # Configuration files
    ├── 📄 development.yaml              # Development config
    ├── 📄 production.yaml               # Production config
    └── 📄 testing.yaml                  # Testing config
```

## 🧪 Testing Structure

### Test Organization
```
tests/
├── 📁 unit/                             # Unit tests
│   ├── 📁 forecasting/                  # Forecasting unit tests
│   ├── 📁 etl/                          # ETL unit tests
│   ├── 📁 analysis/                     # Analysis unit tests
│   └── 📁 optimization/                 # Optimization unit tests
├── 📁 integration/                      # Integration tests
│   ├── 📁 api/                          # API integration tests
│   ├── 📁 database/                     # Database integration tests
│   └── 📁 external/                     # External service tests
├── 📁 e2e/                              # End-to-end tests
│   ├── 📁 forecasting/                  # Forecasting E2E tests
│   ├── 📁 etl/                          # ETL E2E tests
│   └── 📁 full_system/                  # Full system tests
└── 📁 fixtures/                         # Test fixtures
    ├── 📁 data/                         # Test data
    ├── 📁 models/                       # Test models
    └── 📁 configs/                      # Test configurations
```

## 📚 Documentation Structure

### Documentation Organization
```
docs/
├── 📄 architecture.md                   # System architecture
├── 📄 project_structure.md              # This file
├── 📄 deployment.md                     # Deployment guide
├── 📁 user_guides/                      # User documentation
│   ├── 📄 getting_started.md            # Getting started guide
│   ├── 📄 user_manual.md                # User manual
│   └── 📄 troubleshooting.md            # Troubleshooting guide
├── 📁 technical/                        # Technical documentation
│   ├── 📄 api_reference.md              # API reference
│   ├── 📄 database_schema.md            # Database schema
│   └── 📄 performance_guide.md          # Performance guide
└── 📁 research/                         # Research documentation
    ├── 📄 methodology.md                # Research methodology
    ├── 📄 results.md                    # Research results
    └── 📄 papers/                       # Research papers
```

## 🔄 Migration Notes

### Files Moved During Refactoring
- `simple_forecasting_script_parallel.py` → `forecasting/scripts/`
- `autoformer_gabs/*.py` → `forecasting/scripts/`
- `autoformer_gabs/*.ipynb` → `forecasting/notebooks/`
- `autoformer_gabs/*.png` → `forecasting/error_analysis/`
- `Error_gabs/*.png` → `forecasting/error_analysis/`
- `Shap_gabs/*.png` → `forecasting/error_analysis/`
- `LIME_gabs/*.png` → `forecasting/error_analysis/`
- `PDP_gabs/*.png` → `forecasting/error_analysis/`
- `DORC_Trails_and_trainings/TFT_MODEL_training/*.ipynb` → `forecasting/trials/phase2_advanced_models/`
- `DORC_Trails_and_trainings/EIA_TRIAL/*.ipynb` → `forecasting/trials/phase1_initial_exploration/`
- `DORC_PROCESS_OPTIMIZER/DORC_ETL/*` → `agents/etl/`
- `DORC_PROCESS_OPTIMIZER/*.py` → `agents/etl/`
- `DORC_PROCESS_OPTIMIZER/processed*` → `agents/etl/data/`

### Directory Cleanup
The following directories can be safely removed after verification:
- `autoformer_gabs/` (files moved to forecasting/)
- `Error_gabs/` (files moved to forecasting/error_analysis/)
- `Shap_gabs/` (files moved to forecasting/error_analysis/)
- `LIME_gabs/` (files moved to forecasting/error_analysis/)
- `PDP_gabs/` (files moved to forecasting/error_analysis/)
- `DORC_Trails_and_trainings/` (files moved to forecasting/trials/)
- `DORC_PROCESS_OPTIMIZER/` (files moved to agents/etl/)

## 🎯 Next Steps

### Immediate Actions
1. **Verify File Moves**: Ensure all files are in correct locations
2. **Update Import Paths**: Update any hardcoded paths in scripts
3. **Test Functionality**: Verify all moved scripts still work
4. **Clean Up**: Remove empty directories after verification

### Future Development
1. **Complete ETL Agent**: Finish ETL agent development
2. **Start Analysis Agent**: Begin analysis agent development
3. **System Integration**: Integrate all agents
4. **Production Deployment**: Deploy complete system

---

**Project Structure Status**: ✅ Completed
**Last Updated**: January 2024
**Next Review**: Monthly


