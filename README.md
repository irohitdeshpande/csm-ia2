# 📰 Newsstand Inventory Simulation

A **Monte Carlo & Discrete-Event Simulation** application for optimizing newspaper inventory management using **SimPy**, **Streamlit**, and **Plotly**.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Installation](#-installation)
- [How to Use](#-how-to-use)
- [Simulation Methodology](#-simulation-methodology)
- [Performance Metrics](#-performance-metrics)
- [Calculation Formulas](#-calculation-formulas)
- [Optimization Analysis](#-optimization-analysis)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)

---

## 📖 Problem Statement

### Business Context
A newsstand owner faces the classic **inventory optimization problem**: determining the optimal number of newspapers to purchase daily to maximize profit while minimizing waste and lost sales.

### Key Challenges
1. **Uncertain Demand**: Daily newspaper demand varies based on day type (Good, Fair, Poor)
2. **Perishability**: Unsold newspapers have minimal salvage value
3. **Opportunity Cost**: Understocking results in lost profit from unmet demand
4. **Bundle Constraints**: Newspapers can only be purchased in fixed bundle sizes

### Decision Variables
- **Number of papers to purchase daily** (must be a multiple of bundle size)

### Objectives
- **Maximize daily profit** by balancing inventory costs, sales revenue, salvage value, and opportunity costs
- **Minimize waste** from unsold inventory
- **Reduce stockouts** that lead to lost sales

### Constraints
- Papers must be purchased in multiples of bundle size
- All unsold papers can be sold as scrap at reduced price
- Demand varies probabilistically based on day type

---

## 🎯 Features

### SimPy Discrete-Event Simulation
- **Process-Based Modeling**: Newsstand operations modeled as continuous processes
- **Proper Time Management**: SimPy environment handles event scheduling and time progression
- **Event-Driven Architecture**: Each day is a discrete event in the simulation timeline
- **Extensible Framework**: Easy to add complexity (suppliers, customers, resources)

### User-Configurable Parameters
- **Cost Parameters**: Cost price (₹), selling price (₹), and scrap price (₹) per newspaper
- **Bundle Parameters**: Bundle size and number of papers to purchase daily
- **Day Type Probabilities**: Probability distribution for Good, Fair, and Poor days (auto-adjusting)
- **Demand Distribution**: Min/max demand ranges for each day type
- **Simulation Settings**: Number of days (1-365) and animation speed (0.5-1.0 seconds)

### Real-Time Visualizations
1. **Daily Profit Chart**: Track profit trends over time with average line
2. **Demand vs Papers Bought**: Compare actual demand against inventory
3. **Day Type Distribution**: Pie chart showing observed distribution of day types
4. **Cumulative Profit**: Running total of profits over the simulation period
5. **Smooth Animations**: Charts update every 5 days with 500ms transitions for flicker-free viewing

### Comprehensive Analytics
- Random digit generation for day types and demand
- Cumulative probability tables with random digit range assignments
- Detailed simulation table showing:
  - Type of Day
  - Demand
  - Revenue from Sales
  - Lost Profit from Excess Demand
  - Daily Profit
  - Salvage from Scrap Sales
  
### Optimization Engine
- Automatically tests multiple purchase quantities
- Finds the optimal number of papers to maximize profit
- Provides comparison charts and improvement metrics

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher

### Quick Start (Windows)

1. **Run the installer:**
```bash
install.bat
```
This will:
- Create a virtual environment
- Install all required packages
- Set up the application

2. **Run the application:**
```bash
run.bat
```

3. **Open your browser:**
The application will automatically open at `http://localhost:8501`

### Manual Installation (Any Platform)

1. **Create virtual environment:**
```bash
python -m venv venv
```

2. **Activate virtual environment:**
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Open your browser:**
The application will automatically open at `http://localhost:8501`

## 📖 How to Use

### Step 1: Configure Parameters

#### Cost Parameters (in Rupees ₹)
- **Cost Price per Newspaper**: Purchase cost (default: ₹3.30)
- **Selling Price per Newspaper**: Retail price (default: ₹5.00)
- **Scrap Price per Newspaper**: Salvage value for unsold papers (default: ₹0.50)

#### Bundle Parameters
- **Bundle Size**: Minimum purchase quantity (default: 10)
- **Number of Papers to Purchase Daily**: Must be multiple of bundle size (default: 70)

#### Day Type Probabilities (Auto-Adjusting)
- **Good Day Probability**: Likelihood of high demand (default: 0.35)
- **Fair Day Probability**: Likelihood of moderate demand (default: 0.45)
- **Poor Day Probability**: Likelihood of low demand (default: 0.20)
- *Note: Changing any two sliders automatically adjusts the third to maintain sum = 1.0*

#### Demand Distribution (by Day Type)
- **Good Day**: Min 40, Max 100 papers
- **Fair Day**: Min 30, Max 80 papers
- **Poor Day**: Min 20, Max 70 papers

#### Simulation Settings
- **Number of Days**: 1 to 365 days (default: 365)
- **Simulation Speed**: 0.5 to 1.0 seconds per day (default: 0.8)
Click the "🚀 Run Simulation" button in the sidebar to start

#### Simulation Settings
- **Number of Days**: 1 to 365 days (default: 365)
- **Simulation Speed**: 0.5 to 1.0 seconds per day (default: 0.8)

### Step 2: Run Simulation
Click the "🚀 Run Simulation" button in the sidebar to start the discrete-event simulation.

### Step 3: Watch Real-Time Updates
The simulation displays live updates including:
- **Progress Bar**: Current day and completion percentage
- **Live Metrics**: 
  - Total Profit (₹)
  - Average Daily Profit (₹)
  - Average Lost Profit (₹)
  - Average Demand (papers)
- **Real-time Charts**: Updated every 5 days for smooth performance
- **Live Data Table**: Last 20 days of simulation results

### Step 4: Analyze Results
After completion, review:
- **Summary Statistics**: Total/average profits, demand statistics, salvage totals
- **Complete Simulation Table**: All days with detailed breakdown
- **Optimization Analysis**: Best purchase quantity recommendation
- **Downloadable CSV**: Export data for further analysis

---

## � Simulation Methodology

### Overview
This application implements a **Monte Carlo simulation** within a **Discrete-Event (SimPy)** framework to model the stochastic nature of newspaper demand and day types.

### Simulation Procedure

#### Phase 1: Initialization
1. **Build Day Type Lookup Table**
   - Convert probability percentages to cumulative ranges (0-99)
   - Map random digits to day types (Good/Fair/Poor)
   
2. **Build Demand Lookup Tables**
   - For each day type, create uniform probability distribution
   - Map random digits (0-99) to demand quantities
   - Ensure all demands are multiples of bundle size

#### Phase 2: Daily Process (SimPy Environment)
For each simulated day:

1. **Generate Day Type**
   - Generate random digit RD₁ ∈ [0, 99]
   - Lookup day type using cumulative probability table
   
2. **Generate Demand**
   - Generate random digit RD₂ ∈ [0, 99]
   - Lookup demand quantity for the determined day type
   
3. **Calculate Transactions**
   - Papers Sold = min(Demand, Papers Bought)
   - Papers Unsold = max(0, Papers Bought - Demand)
   - Excess Demand = max(0, Demand - Papers Bought)
   
4. **Calculate Daily Profit**
   - Revenue = Papers Sold × Selling Price
   - Cost = Papers Bought × Cost Price
   - Salvage = Papers Unsold × Scrap Price
   - Lost Profit = Excess Demand × (Selling Price - Cost Price)
   - **Daily Profit = Revenue - Cost - Lost Profit + Salvage**
   
5. **Record Results**
   - Store all metrics for the day
   - Update cumulative statistics
   
6. **Advance Time**
   - `yield env.timeout(1)` advances simulation by 1 day

#### Phase 3: Optimization
After simulation completes:

1. **Define Test Range**
   - Minimum Test = 80% of observed minimum demand
   - Maximum Test = 120% of observed maximum demand
   - Step = Bundle Size
   
2. **Test Each Quantity**
   - For each test quantity Q:
     - Apply Q to all simulated demand scenarios
     - Calculate average daily profit
   
3. **Identify Optimal**
   - Select quantity with maximum average profit
   - Calculate improvement vs. current quantity
   - Display visual comparison

### Probability Distribution Method

#### Day Type Assignment
```
Cumulative Probability → Random Digit Range
P(Good) = 0.35 → Digits 00-34
P(Fair) = 0.45 → Digits 35-79  
P(Poor) = 0.20 → Digits 80-99
```

#### Demand Assignment (Uniform Distribution)
For each day type, demands are equally likely:
```
If Good Day Demand Range = [40, 50, 60, 70, 80, 90, 100]
Then P(each demand) = 1/7 ≈ 0.143
Digit ranges assigned proportionally (e.g., 40→00-13, 50→14-27, etc.)
```

---

## 📊 Performance Metrics

### Primary Metrics

#### 1. Daily Profit (₹)
**Formula**: `Revenue - Cost - Lost Profit + Salvage`

**Components**:
- Revenue from Sales
- Cost of Papers Bought
- Lost Profit from Stockouts
- Salvage from Unsold Papers

**Interpretation**: Net profit/loss for each day

#### 2. Total Profit (₹)
**Formula**: `Σ(Daily Profit)` for all simulation days

**Interpretation**: Cumulative profit over entire simulation period

#### 3. Average Daily Profit (₹)
**Formula**: `Total Profit / Number of Days`

**Interpretation**: Expected profit per day under given parameters

#### 4. Standard Deviation of Profit (₹)
**Formula**: `σ = √[Σ(Daily Profit - Avg Profit)² / (n-1)]`

**Interpretation**: Variability/risk in daily profits

### Secondary Metrics

#### 5. Total Lost Profit (₹)
**Formula**: `Σ(Excess Demand × Unit Profit Margin)` across all days

**Interpretation**: Revenue lost due to stockouts

#### 6. Average Lost Profit (₹/day)
**Formula**: `Total Lost Profit / Number of Days`

**Interpretation**: Average opportunity cost per day

#### 7. Total Salvage Value (₹)
**Formula**: `Σ(Unsold Papers × Scrap Price)` across all days

**Interpretation**: Revenue recovered from unsold inventory

#### 8. Average Salvage (₹/day)
**Formula**: `Total Salvage / Number of Days`

**Interpretation**: Average waste recovery per day

#### 9. Average Demand (papers)
**Formula**: `Σ(Daily Demand) / Number of Days`

**Interpretation**: Mean customer demand

#### 10. Demand Range
- **Maximum Demand**: Highest demand observed
- **Minimum Demand**: Lowest demand observed

**Interpretation**: Variability in customer demand

### Optimization Metrics

#### 11. Optimal Purchase Quantity (papers)
**Definition**: Quantity that maximizes average daily profit in tested range

**Calculation**: `argmax Q ∈ [Q_min, Q_max] { E[Profit(Q)] }`

#### 12. Profit Improvement (₹/day)
**Formula**: `Optimal Avg Profit - Current Avg Profit`

**Color Coding**:
- Green ↑: Positive improvement (optimal > current)
- Red ↓: Negative improvement (optimal < current)

#### 13. Annual Impact (₹/year)
**Formula**: `Profit Improvement × 365 days`

**Interpretation**: Projected annual financial impact of optimization

---

## 🧮 Calculation Formulas

### Core Profit Formula

```
Daily Profit = Revenue - Cost - Lost Profit + Salvage
```

### Detailed Component Formulas

#### 1. Revenue from Sales (₹)
```
Revenue = Papers Sold × Selling Price
where Papers Sold = min(Demand, Papers Bought)
```

#### 2. Cost of Papers (₹)
```
Cost = Papers Bought × Cost Price
```

#### 3. Salvage Value (₹)
```
Salvage = Papers Unsold × Scrap Price
where Papers Unsold = max(0, Papers Bought - Demand)
```

#### 4. Lost Profit from Stockouts (₹)
```
Lost Profit = Excess Demand × Unit Profit Margin
where:
  Excess Demand = max(0, Demand - Papers Bought)
  Unit Profit Margin = Selling Price - Cost Price
```

### Example Calculation

**Given**:
- Cost Price = ₹3.30
- Selling Price = ₹5.00
- Scrap Price = ₹0.50
- Papers Bought = 70
- Demand = 85

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

### Probability Calculations

#### Cumulative Probability to Range
```
For probability P and total range [0, 99]:
Start Range = floor(Cumulative Probability × 100)
End Range = floor((Cumulative Probability + P) × 100) - 1
```

#### Uniform Demand Distribution
```
For n possible demands in range [d_min, d_max]:
P(each demand) = 1 / n
Range width per demand ≈ 100 / n digits
```

---

## 🎯 Optimization Analysis

## � Optimization Analysis

### Methodology
The optimization engine performs **post-simulation analysis** to find the purchase quantity that would have yielded maximum profit given the observed demand patterns.

### Optimization Process

1. **Define Search Space**
   ```
   Min Test Quantity = max(Bundle Size, 0.8 × Min Observed Demand)
   Max Test Quantity = 1.2 × Max Observed Demand + 2 × Bundle Size
   Step = Bundle Size
   ```

2. **Evaluate Each Quantity**
   - For each test quantity Q:
     - Apply Q to all simulated days
     - Recalculate daily profits using actual demands
     - Compute average daily profit
   
3. **Select Optimal**
   ```
   Optimal Quantity = argmax { Average Daily Profit(Q) }
   ```

4. **Calculate Improvement**
   ```
   Daily Improvement = Optimal Profit - Current Profit
   Annual Improvement = Daily Improvement × 365
   Percentage Improvement = (Daily Improvement / Current Profit) × 100%
   ```

### Visualization
- **Profit Curve**: Shows profit vs. quantity relationship
- **Green Vertical Line**: Marks optimal quantity
- **Red Vertical Line**: Marks current quantity
- **Comparison**: Visual gap between optimal and current

### Interpretation
- **Optimal = Current**: Already at best quantity for observed demand
- **Optimal > Current**: Understocking - increase purchase quantity
- **Optimal < Current**: Overstocking - decrease purchase quantity

---

## 📝 Technical Details

### Technology Stack
- **Python 3.8+**: Core programming language
- **SimPy 4.x**: Discrete-event simulation framework
- **Streamlit 1.x**: Interactive web application framework
- **Pandas**: Data manipulation and tabular analysis
- **NumPy**: Numerical computations and random number generation
- **Plotly**: Interactive, animated visualizations

### Architecture

#### Simulation Engine (`NewsstandSimulation` Class)
- **Environment**: SimPy discrete-event environment
- **Process**: Generator function modeling daily operations
- **Lookup Tables**: Pre-computed probability mappings
- **Data Collection**: Real-time recording of all transactions

#### Key Components
1. **Day Type Lookup Builder**: Converts probabilities to digit ranges
2. **Demand Lookup Builder**: Creates uniform distributions per day type
3. **Day Type Determiner**: Maps random digits to day types
4. **Demand Generator**: Maps random digits to demand quantities
5. **Profit Calculator**: Computes all financial metrics
6. **Optimization Engine**: Post-simulation quantity analysis

### Simulation Characteristics

| Characteristic | Value |
|---|---|
| Simulation Type | Monte Carlo + Discrete-Event |
| Time Model | Discrete (daily steps) |
| Randomness | Pseudo-random (NumPy) |
| Event Scheduling | SimPy process-based |
| Data Structure | Pandas DataFrame |
| Visualization | Plotly (interactive) |
| Update Frequency | Every 5 days |
| Transition Duration | 500ms (smooth) |

### Performance Considerations
- **Chart Updates**: Throttled to every 5 days to prevent UI lag
- **Smooth Transitions**: 500ms animation duration prevents flickering
- **Memory Efficient**: Uses generators for large simulations
- **Type Safety**: Type hints for better code maintainability

---

## 🎨 Visualization Guide

### 1. Daily Profit Over Time
**Type**: Line chart with area fill
- **X-Axis**: Day number
- **Y-Axis**: Profit (₹)
- **Green Line**: Daily profit values
- **Red Dashed Line**: Average profit
- **Purpose**: Identify trends and volatility

### 2. Daily Demand vs Papers Bought
**Type**: Scatter plot with reference line
- **X-Axis**: Day number
- **Y-Axis**: Number of papers
- **Color Scale**: Demand intensity (Viridis)
- **Orange Dashed Line**: Purchase quantity
- **Purpose**: Visualize over/understocking patterns

### 3. Distribution of Day Types
**Type**: Donut chart
- **Segments**: Good (green), Fair (orange), Poor (red)
- **Values**: Percentage and count
- **Purpose**: Validate probability distributions

### 4. Cumulative Profit Over Time
**Type**: Area chart
- **X-Axis**: Day number
- **Y-Axis**: Cumulative profit (₹)
- **Purple Line**: Running total
- **Purpose**: Track overall financial performance

### 5. Profit vs Papers Purchased
**Type**: Line chart with markers
- **X-Axis**: Purchase quantity
- **Y-Axis**: Average daily profit (₹)
- **Blue Line**: Profit curve
- **Green Dashed Line**: Optimal quantity
- **Red Dashed Line**: Current quantity
- **Purpose**: Identify optimal purchase point

---

## � Best Practices & Tips

### Parameter Configuration
1. **Realistic Probabilities**: Ensure day type probabilities sum to 1.0 (auto-adjusted)
2. **Demand Ranges**: Base on historical data or market research
3. **Cost Structure**: Verify selling price > cost price > scrap price
4. **Bundle Size**: Match actual supplier constraints

### Simulation Settings
1. **Longer Simulations**: 300-365 days provide more stable results
2. **Speed Adjustment**: Use 0.5-0.7 seconds for detailed observation
3. **Multiple Runs**: Test different scenarios to understand sensitivity

### Analysis Approach
1. **Check Day Type Distribution**: Should match configured probabilities
2. **Review Average Demand**: Should fall within configured ranges
3. **Analyze Profit Variance**: High variance indicates unstable demand
4. **Compare Optimization**: Large improvement suggests suboptimal current quantity

### Exporting Results
1. **Download CSV**: Click download button after simulation
2. **External Analysis**: Use Excel/Python for advanced analytics
3. **Documentation**: Record parameters used for each scenario

---

### Application won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (must be 3.8+)

### Simulation runs slowly
- Reduce the number of simulation days
- Increase the simulation speed slider to 0

### Charts not updating
- This is normal - charts update every 5 days to improve performance
- They will update at the end of the simulation

---

## 🔧 Troubleshooting

### Application Issues

#### Application won't start
**Symptoms**: Error when running `streamlit run app.py`
**Solutions**:
1. Verify Python version: `python --version` (must be 3.8+)
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check virtual environment activation
4. Ensure all required files are present

#### Slow Performance
**Symptoms**: Simulation takes too long
**Solutions**:
1. Reduce number of simulation days (try 100-200 initially)
2. Increase simulation speed to 0.5 seconds
3. Close other browser tabs
4. Charts update every 5 days (this is intentional for performance)

#### Charts Not Updating
**Symptoms**: Visualizations appear frozen
**Solutions**:
- This is normal behavior - charts update every 5 days
- Charts will update at simulation completion
- Final results show all data

#### Probability Sliders Not Adjusting
**Symptoms**: Third slider doesn't auto-adjust
**Solutions**:
1. Ensure you're changing sliders one at a time
2. Wait for page to rerun after each change
3. Probabilities should sum to 1.0 automatically

### Data Issues

#### Unrealistic Results
**Symptoms**: Profits seem too high/low
**Solutions**:
1. Verify cost parameters are in correct units (₹)
2. Check demand ranges are reasonable
3. Ensure selling price > cost price > scrap price
4. Review day type probabilities

#### Optimization Shows No Improvement
**Symptoms**: Optimal = Current quantity
**Solutions**:
- This indicates you're already at optimal quantity
- Try different demand distributions
- Run longer simulations for more data

---

## 📚 Example Scenarios

### Scenario 1: Conservative Strategy
**Configuration**:
- Cost Price: ₹3.30, Selling Price: ₹5.00, Scrap: ₹0.50
- Bundle: 10, Purchase: 50 papers/day
- Good (35%): 40-100, Fair (45%): 30-80, Poor (20%): 20-70

**Expected Behavior**:
- Lower risk due to conservative purchase quantity
- Some lost sales on Good days
- Minimal waste on Poor days
- Stable but potentially suboptimal profit

### Scenario 2: Aggressive Strategy
**Configuration**:
- Same costs as Scenario 1
- Purchase: 90 papers/day (higher)
- Same day type distributions

**Expected Behavior**:
- Higher profit potential on Good days
- Significant waste on Poor days
- Higher variance in daily profits
- Optimization likely suggests reduction

### Scenario 3: Balanced Strategy
**Configuration**:
- Same costs as Scenario 1
- Purchase: 70 papers/day (medium)
- Same day type distributions

**Expected Behavior**:
- Balance between stockouts and waste
- Moderate profit with acceptable risk
- Likely near-optimal for given distributions

---

## 📄 Project Structure

```
CSM-IA/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── CODE_EXPLANATION.md    # Detailed code documentation
├── QUICKSTART.md          # Quick start guide
├── install.bat            # Windows installer script
├── run.bat                # Windows run script
└── .gitignore             # Git ignore rules
```

---

## 📦 Dependencies

```
streamlit>=1.28.0          # Web application framework
simpy>=4.1.0               # Discrete-event simulation
pandas>=2.1.0              # Data manipulation
numpy>=1.24.0              # Numerical computing
plotly>=5.17.0             # Interactive visualizations
```

---

## 🎓 Educational Value

### Learning Objectives
This simulation helps understand:
1. **Inventory Management**: Newsvendor problem fundamentals
2. **Stochastic Modeling**: Dealing with uncertain demand
3. **Monte Carlo Methods**: Using randomness for predictions
4. **Discrete-Event Simulation**: Modeling time-dependent systems
5. **Optimization**: Finding best decisions under uncertainty
6. **Trade-off Analysis**: Balancing costs, revenue, and risk

### Applications
- Supply chain management education
- Operations research coursework
- Business analytics training
- Decision science demonstrations
- Risk management studies

---

## 📝 Technical Details

### Technology Stack
- **SimPy**: Discrete-event simulation framework
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations and random number generation
- **Plotly**: Interactive visualizations

### Simulation Method
Process-based discrete-event simulation using SimPy:
- **SimPy Environment**: Manages simulation time and event scheduling
- **Process Definition**: Newsstand modeled as a continuous process
- **Event Generation**: Random events for day types and demand
- **Time Progression**: `yield env.timeout(1)` advances simulation by one day
- **Data Collection**: Results captured at each time step
- **Statistical Analysis**: Comprehensive profit and demand analysis

## 📄 License

This project is created for educational and business analysis purposes.

## 🤝 Contributing

Feel free to fork this project and customize it for your specific needs!

## 📧 Support

For questions or issues, please open an issue in the repository.

---

**Happy Simulating! 📊📰**
#   c s m - i a 2 
 
 #   c s m - i a 2 
 
 
