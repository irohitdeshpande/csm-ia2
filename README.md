# 📰 Newsstand Inventory Simulation

A **Monte Carlo & Discrete-Event Simulation** application for optimizing newspaper inventory management using **SimPy**, **Streamlit**, and **Plotly**.

---

## 📋 Table of Contents

- [Problem Definition](#-problem-definition)
- [Simulation Model](#-simulation-model)
- [How to Run](#-how-to-run)
- [Features](#-features)
- [User Guide](#-user-guide)
- [Calculation Formulas](#-calculation-formulas)
- [Optimization Analysis](#-optimization-analysis)
- [Technical Stack](#-technical-stack)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Problem Definition

### Business Context

A newsstand owner faces the classic **inventory optimization problem**: determining the optimal number of newspapers to purchase daily to maximize profit while minimizing waste and lost sales.

### Key Challenges

1. **Uncertain Demand**: Daily newspaper demand varies based on day type (Good, Fair, Poor)
2. **Perishability**: Unsold newspapers have minimal salvage value
3. **Opportunity Cost**: Understocking results in lost profit from unmet demand
4. **Bundle Constraints**: Newspapers can only be purchased in fixed bundle sizes

### Decision Problem

**What is the optimal number of newspapers to purchase daily?**

**Objective**: Maximize profit by balancing:
- Revenue from sales
- Cost of papers purchased
- Salvage value from unsold papers
- Lost profit from unmet demand (opportunity cost)

**Constraints**:
- Papers must be purchased in multiples of bundle size
- All unsold papers can be sold as scrap at reduced price
- Demand varies probabilistically by day type

---

## 📊 Simulation Model

### Overview

This application implements a **Monte Carlo Simulation** within a **Discrete-Event (SimPy) Framework** to model the stochastic nature of newspaper demand and day types.

### Simulation Architecture

#### Phase 1: Initialization

The simulation creates lookup tables for day types and demand distributions:

1. **Day Type Lookup Table**
   - Converts probability percentages to cumulative ranges (0-99)
   - Maps random digits to day types: Good, Fair, or Poor

2. **Demand Lookup Tables**
   - For each day type, creates uniform probability distribution
   - Maps random digits (0-99) to demand quantities
   - Ensures all demands are multiples of bundle size

#### Phase 2: Daily Process (SimPy Environment)

For each simulated day:

1. **Generate Day Type**
   - Generate random digit RD₁ ∈ [0, 99]
   - Lookup day type using cumulative probability table

2. **Generate Demand**
   - Generate random digit RD₂ ∈ [0, 99]
   - Lookup demand quantity for the determined day type

3. **Calculate Daily Transactions**
   - `Papers Sold = min(Demand, Papers Bought)`
   - `Papers Unsold = max(0, Papers Bought - Demand)`
   - `Excess Demand = max(0, Demand - Papers Bought)`

4. **Calculate Daily Profit**
   ```
   Revenue = Papers Sold × Selling Price
   Cost = Papers Bought × Cost Price
   Salvage = Papers Unsold × Scrap Price
   Lost Profit = Excess Demand × (Selling Price - Cost Price)
   
   Daily Profit = Revenue - Cost - Lost Profit + Salvage
   ```

5. **Record Results**
   - Store all metrics for the day
   - Update cumulative statistics

6. **Advance Time**
   - `yield env.timeout(1)` advances simulation by 1 day

#### Phase 3: Optimization

After simulation completes:

1. **Define Search Space**
   - Min Test: 80% of observed minimum demand
   - Max Test: 120% of observed maximum demand
   - Step: Bundle Size

2. **Test Each Quantity**
   - For each test quantity Q: Apply Q to all simulated demand scenarios
   - Recalculate daily profits
   - Compute average daily profit

3. **Identify Optimal**
   - Select quantity with maximum average profit
   - Calculate improvement vs. current quantity
   - Display visual comparison

### Probability Distribution Method

#### Day Type Assignment

```
Good Day:    Probability 0.35 → Random Digits 00-34
Fair Day:    Probability 0.45 → Random Digits 35-79
Poor Day:    Probability 0.20 → Random Digits 80-99
```

#### Demand Assignment (Uniform Distribution)

For each day type, demands are equally likely across the configured range:

```
Example: Good Day Demand Range = [40, 50, 60, 70, 80, 90, 100]
P(each demand) = 1/7 ≈ 0.143
Digit ranges assigned proportionally (e.g., 40→00-13, 50→14-27, etc.)
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Quick Start

#### Option 1: Windows Batch Scripts

1. **Run the installer** (creates virtual environment and installs dependencies):
   ```bash
   install.bat
   ```

2. **Run the application**:
   ```bash
   run.bat
   ```

3. **Open your browser**: Navigate to `http://localhost:8501`

#### Option 2: Manual Installation (Any Platform)

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**: Navigate to `http://localhost:8501`

### Required Dependencies

```
streamlit>=1.28.0          # Web application framework
simpy>=4.1.0               # Discrete-event simulation
pandas>=2.1.0              # Data manipulation
numpy>=1.24.0              # Numerical computing
plotly>=5.17.0             # Interactive visualizations
```

---

## ✨ Features

### Real-Time Discrete-Event Simulation

- **Process-Based Modeling**: Newsstand operations modeled as continuous SimPy processes
- **Proper Time Management**: SimPy environment handles event scheduling
- **Event-Driven Architecture**: Each day is a discrete event in simulation timeline
- **Extensible Framework**: Easy to add complexity (suppliers, customers, resources)

### User-Configurable Parameters

- **Cost Parameters**: Cost price, selling price, and scrap price per newspaper (₹)
- **Bundle Parameters**: Bundle size and number of papers to purchase daily
- **Day Type Probabilities**: Probability distribution for Good/Fair/Poor days (auto-adjusting)
- **Demand Distribution**: Min/max demand ranges for each day type
- **Simulation Settings**: Number of days (1-365, default: 20) and animation speed (0.5-1.0 seconds)

### Real-Time Visualizations

1. **Daily Profit Chart**: Track profit trends with average line
2. **Demand vs Papers Bought**: Compare actual demand against inventory
3. **Day Type Distribution**: Pie chart of observed day types
4. **Cumulative Profit**: Running total of profits
5. **Smooth Animations**: Charts update every 3 days with 500ms transitions

### Comprehensive Analytics

- **Random Digit Generation**: For day types and demand
- **Cumulative Probability Tables**: With random digit range assignments
- **Detailed Simulation Table** showing:
  - Type of Day
  - Demand
  - Revenue from Sales
  - Lost Profit from Excess Demand
  - Daily Profit
  - Salvage from Scrap Sales

### Optimization Engine

- Tests multiple purchase quantities automatically
- Identifies optimal quantity to maximize profit
- Provides comparison charts and improvement metrics
- Calculates annual financial impact

---

## 📖 User Guide

### Step 1: Configure Parameters

#### Cost Parameters (in Rupees ₹)

- **Cost Price**: Purchase cost per newspaper (default: ₹3.30)
- **Selling Price**: Retail price per newspaper (default: ₹5.00)
- **Scrap Price**: Salvage value for unsold papers (default: ₹0.50)

**Best Practice**: Ensure `Selling Price > Cost Price > Scrap Price`

#### Bundle Parameters

- **Bundle Size**: Minimum purchase quantity (default: 10)
- **Number of Papers to Purchase Daily**: Must be multiple of bundle size (default: 70)

#### Day Type Probabilities

- **Good Day Probability** (default: 0.35)
- **Fair Day Probability** (default: 0.45)
- **Poor Day Probability** (default: 0.20)

**Note**: Changing any two sliders automatically adjusts the third to maintain sum = 1.0

#### Demand Distribution

Configure min/max demand for each day type:

| Day Type | Min | Max |
|----------|-----|-----|
| Good Day | 40 | 100 |
| Fair Day | 30 | 80 |
| Poor Day | 20 | 70 |

#### Simulation Settings

- **Number of Days**: 1 to 365 days (default: 20)
- **Simulation Speed**: 0.5 to 1.0 seconds per day (default: 0.5)

### Step 2: Run Simulation

1. Configure all parameters in the sidebar
2. Click the **"🚀 Run Simulation"** button
3. Watch real-time updates as the simulation progresses

### Step 3: Monitor Real-Time Updates

During simulation, you'll see:

- **Progress Bar**: Current day and completion percentage
- **Live Metrics**:
  - Total Profit (₹)
  - Average Daily Profit (₹)
  - Average Lost Profit (₹)
  - Average Demand (papers)
- **Real-Time Charts**: Updated every 3 days
- **Live Data Table**: Last 20 days of results

### Step 4: Analyze Results

After completion, review:

- **Summary Statistics**: Total/average profits, demand statistics, salvage totals
- **Complete Simulation Table**: All days with detailed breakdown
- **Optimization Analysis**: Best purchase quantity recommendation
- **Downloadable CSV**: Export data for further analysis

---

## 🧮 Calculation Formulas

### Core Profit Formula

```
Daily Profit = Revenue - Cost - Lost Profit + Salvage
```

### Component Formulas

**Revenue from Sales (₹)**
```
Revenue = Papers Sold × Selling Price
where Papers Sold = min(Demand, Papers Bought)
```

**Cost of Papers (₹)**
```
Cost = Papers Bought × Cost Price
```

**Salvage Value (₹)**
```
Salvage = Papers Unsold × Scrap Price
where Papers Unsold = max(0, Papers Bought - Demand)
```

**Lost Profit from Stockouts (₹)**
```
Lost Profit = Excess Demand × Unit Profit Margin
where:
  Excess Demand = max(0, Demand - Papers Bought)
  Unit Profit Margin = Selling Price - Cost Price
```

### Example Calculation

**Given**:
- Cost Price: ₹3.30
- Selling Price: ₹5.00
- Scrap Price: ₹0.50
- Papers Bought: 70
- Demand: 85

**Calculation**:
```
Papers Sold = min(85, 70) = 70
Papers Unsold = max(0, 70 - 85) = 0
Excess Demand = max(0, 85 - 70) = 15

Revenue = 70 × ₹5.00 = ₹350.00
Cost = 70 × ₹3.30 = ₹231.00
Salvage = 0 × ₹0.50 = ₹0.00
Lost Profit = 15 × (₹5.00 - ₹3.30) = 15 × ₹1.70 = ₹25.50

Daily Profit = ₹350.00 - ₹231.00 - ₹25.50 + ₹0.00 = ₹93.50
```

---

## 🎯 Optimization Analysis

### What is Optimization?

The optimization engine performs **post-simulation analysis** to find the purchase quantity that would have yielded maximum profit given the observed demand patterns from the simulation.

### How It Works

1. **Test All Quantities**: Tests multiple purchase quantities using simulated demands
2. **Calculate Average Profit**: For each quantity, computes average daily profit
3. **Find Best**: Selects the quantity with maximum average profit
4. **Compare**: Shows improvement vs. current quantity

### Interpretation

| Scenario | Meaning | Action |
|----------|---------|--------|
| Optimal = Current | Already at best quantity | No change needed |
| Optimal > Current | Understocking | Increase purchase quantity |
| Optimal < Current | Overstocking | Decrease purchase quantity |

### Metrics

- **Optimal Quantity**: Purchase quantity that maximizes profit
- **Profit Improvement**: Expected daily profit increase (₹/day)
- **Annual Impact**: Projected annual financial improvement (₹/year)

### Visualization

The optimization chart shows:
- **Blue Line**: Profit curve across all tested quantities
- **Green Dashed Line**: Marks optimal quantity
- **Red Dashed Line**: Marks current quantity
- **Visual Gap**: Difference between optimal and current choices

---

## 📊 Performance Metrics

### Primary Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| Daily Profit | Revenue - Cost - Lost Profit + Salvage | Net profit/loss for each day |
| Total Profit | Σ(Daily Profit) | Cumulative profit over entire simulation |
| Average Daily Profit | Total Profit / Number of Days | Expected profit per day |
| Std Dev of Profit | σ = √[Σ(Daily Profit - Avg)² / (n-1)] | Variability/risk in daily profits |

### Secondary Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| Total Lost Profit | Σ(Excess Demand × Unit Margin) | Revenue lost due to stockouts |
| Average Lost Profit | Total Lost Profit / Number of Days | Average opportunity cost per day |
| Total Salvage | Σ(Unsold Papers × Scrap Price) | Revenue recovered from waste |
| Average Salvage | Total Salvage / Number of Days | Average waste recovery per day |
| Average Demand | Σ(Daily Demand) / Number of Days | Mean customer demand |
| Demand Range | Max - Min Demand | Variability in customer demand |

---

## 🎨 Dashboard Visualizations

### 1. Daily Profit Over Time
- **Type**: Line chart with area fill
- **Shows**: Profit trends and volatility
- **Colors**: Green line with red average reference

### 2. Daily Demand vs Papers Bought
- **Type**: Scatter plot with reference line
- **Shows**: Over/understocking patterns
- **Colors**: Viridis color scale for demand intensity

### 3. Distribution of Day Types
- **Type**: Donut chart
- **Shows**: Observed distribution of day types
- **Colors**: Green (Good), Orange (Fair), Red (Poor)

### 4. Cumulative Profit Over Time
- **Type**: Area chart
- **Shows**: Running total of profits
- **Colors**: Purple line with light purple fill

### 5. Profit vs Papers Purchased
- **Type**: Line chart with markers
- **Shows**: Relationship between quantity and profit
- **Highlights**: Optimal (green) and current (red) quantities

---

## 🔧 Technical Stack

### Technologies

- **SimPy 4.x**: Discrete-event simulation framework
- **Streamlit 1.x**: Interactive web application framework
- **Pandas**: Data manipulation and tabular analysis
- **NumPy**: Numerical computations and random number generation
- **Plotly**: Interactive, animated visualizations
- **Python 3.8+**: Core programming language

### Simulation Characteristics

| Characteristic | Value |
|---|---|
| Simulation Type | Monte Carlo + Discrete-Event |
| Time Model | Discrete (daily steps) |
| Randomness | Pseudo-random (NumPy) |
| Event Scheduling | SimPy process-based |
| Update Frequency | Every 3 days |
| Animation Duration | 500ms (smooth transitions) |

### Code Architecture

- **NewsstandSimulation Class**: Core simulation engine
  - `_build_day_type_lookup()`: Creates day type probability table
  - `_build_demand_lookup()`: Creates demand distribution tables
  - `determine_day_type()`: Maps random digits to day types
  - `generate_demand()`: Maps random digits to demand
  - `calculate_profit()`: Computes daily financial metrics
  - `newsstand_process()`: Main SimPy process

- **Helper Functions**:
  - `create_cumulative_prob_table()`: Displays day type probabilities
  - `create_demand_prob_table()`: Displays demand distributions
  - `calculate_profit_for_quantity()`: Tests optimization quantities

---

## ❓ Troubleshooting

### Application Won't Start

**Problem**: Error when running `streamlit run app.py`

**Solutions**:
1. Verify Python version: `python --version` (must be 3.8+)
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Ensure virtual environment is activated
4. Check that all required files are present

### Slow Performance

**Problem**: Simulation takes too long

**Solutions**:
1. Reduce number of simulation days (try 100-200 initially)
2. Decrease simulation speed to 0.5 seconds
3. Close other browser tabs
4. Charts intentionally update every 3 days for performance

### Charts Not Updating

**Problem**: Visualizations appear frozen

**Solutions**:
- This is normal behavior - charts update every 3 days
- Charts will update at simulation completion
- Final results show all data

### Probability Sliders Not Auto-Adjusting

**Problem**: Third probability slider doesn't automatically adjust

**Solutions**:
1. Change sliders one at a time
2. Wait for page to rerun after each change
3. Probabilities should sum to 1.0 automatically

### Unrealistic Results

**Problem**: Profits seem too high/low

**Solutions**:
1. Verify cost parameters are in correct units (₹)
2. Check demand ranges are reasonable and realistic
3. Ensure `Selling Price > Cost Price > Scrap Price`
4. Review day type probabilities match market conditions

---

## 📚 Example Scenarios

### Scenario 1: Conservative Strategy

**Configuration**:
- Cost: ₹3.30, Selling: ₹5.00, Scrap: ₹0.50
- Bundle: 10, Purchase: 50 papers/day
- Good (35%): 40-100, Fair (45%): 30-80, Poor (20%): 20-70

**Expected Result**: Lower risk, some lost sales on Good days, minimal waste

### Scenario 2: Aggressive Strategy

**Configuration**:
- Same costs as Scenario 1
- Purchase: 90 papers/day (higher)
- Same day type distributions

**Expected Result**: Higher profit potential, significant waste on Poor days, high variance

### Scenario 3: Balanced Strategy

**Configuration**:
- Same costs as Scenario 1
- Purchase: 70 papers/day (medium)
- Same day type distributions

**Expected Result**: Balance between stockouts and waste, likely near-optimal

---

## 💡 Best Practices

### Parameter Configuration

1. **Probabilities**: Ensure day type probabilities sum to 1.0 (auto-adjusted)
2. **Demand Ranges**: Base on historical data or market research
3. **Cost Structure**: Verify `Selling Price > Cost Price > Scrap Price`
4. **Bundle Size**: Match actual supplier constraints

### Simulation Settings

1. **Longer Simulations**: Use 300-365 days for stable results
2. **Speed Adjustment**: Use 0.5-0.7 seconds for detailed observation
3. **Multiple Runs**: Test different scenarios to understand sensitivity

### Analysis Approach

1. Check Day Type Distribution matches configured probabilities
2. Review Average Demand falls within configured ranges
3. Analyze Profit Variance for demand stability
4. Compare Optimization to identify suboptimal quantities

### Exporting Results

1. Click the **"📥 Download Simulation Data (CSV)"** button
2. Use Excel or Python for advanced analytics
3. Record parameters used for each scenario

---

## 📁 Project Structure

```
newsstand-simulation/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── install.bat            # Windows installer script
├── run.bat                # Windows run script
└── .gitignore             # Git ignore rules
```

---

## 🎓 Learning Objectives

This simulation helps understand:

1. **Inventory Management**: Newsvendor problem fundamentals
2. **Stochastic Modeling**: Dealing with uncertain demand
3. **Monte Carlo Methods**: Using randomness for predictions
4. **Discrete-Event Simulation**: Modeling time-dependent systems
5. **Optimization**: Finding best decisions under uncertainty
6. **Trade-off Analysis**: Balancing costs, revenue, and risk

---

## 📝 License

This project is created for educational and business analysis purposes.

---

## 🤝 Contributing

Feel free to fork this project and customize it for your specific needs!

---

## 📧 Support

For questions or issues, please open an issue in the repository.

---

**Happy Simulating! 📊📰**
