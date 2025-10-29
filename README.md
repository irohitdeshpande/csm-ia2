# 📰 Newsstand Inventory Simulation with SimPy

An interactive **discrete-event simulation** for optimizing newspaper inventory management using **SimPy**, Streamlit, and Plotly.

## 🎯 Features

### SimPy Discrete-Event Simulation
- **Process-Based Modeling**: Newsstand operations modeled as continuous processes
- **Proper Time Management**: SimPy environment handles event scheduling and time progression
- **Event-Driven Architecture**: Each day is a discrete event in the simulation timeline
- **Extensible Framework**: Easy to add complexity (suppliers, customers, resources)

### User-Configurable Parameters
- **Cost Parameters**: Cost price, selling price, and scrap price per newspaper
- **Bundle Parameters**: Bundle size and number of papers to purchase daily
- **Day Type Probabilities**: Probability distribution for Good, Fair, and Poor days
- **Demand Distribution**: Min/max demand ranges for each day type
- **Simulation Settings**: Number of days to simulate and animation speed

### Real-Time Visualizations
1. **Daily Profit Chart**: Track profit trends over time with average line
2. **Demand vs Papers Bought**: Compare actual demand against inventory
3. **Day Type Distribution**: Pie chart showing distribution of day types
4. **Cumulative Profit**: Running total of profits over the simulation period

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
streamlit run newsstand_simulation.py
```

5. **Open your browser:**
The application will automatically open at `http://localhost:8501`

## 📖 How to Use

### 1. Configure Parameters
Use the sidebar to set:
- **Cost Parameters**: Define the economic model
- **Bundle Size**: Newspapers can only be bought in multiples of this size
- **Purchase Quantity**: How many papers to buy each day (must be multiple of bundle size)
- **Day Type Probabilities**: Distribution of Good, Fair, and Poor days
- **Demand Ranges**: Expected demand for each day type
- **Simulation Length**: Number of days to simulate

### 2. Run Simulation
Click the "🚀 Run Simulation" button in the sidebar to start

### 3. Watch Real-Time Updates
The simulation will display:
- Progress bar
- Live metrics (Total Profit, Average Profit, Success Rate, Average Demand)
- Real-time updating charts
- Rolling simulation table (last 20 days)

### 4. Analyze Results
After completion, view:
- Complete simulation table
- Summary statistics
- Optimization analysis showing the best purchase quantity
- Downloadable CSV data

## 📊 Understanding the Simulation

### Day Type Generation
- Random digits (0-99) are generated for each day
- Based on cumulative probabilities, days are classified as Good, Fair, or Poor
- Each day type has different demand characteristics

### Demand Generation
- For each day type, demand is randomly selected from the configured range
- Demands are always multiples of the bundle size
- Random digits are recorded for reproducibility

### Profit Calculation
For each day:
- **Revenue** = min(demand, papers_bought) × selling_price
- **Cost** = papers_bought × cost_price
- **Salvage** = max(0, papers_bought - demand) × scrap_price
- **Lost Profit** = max(0, demand - papers_bought) × (selling_price - cost_price)
- **Daily Profit** = Revenue - Cost + Salvage

### Optimization
The system tests various purchase quantities (in bundle size increments) and calculates the expected profit for each quantity based on the simulated demand patterns. The quantity with the highest average daily profit is recommended.

## 🎨 Visualizations Explained

### 1. Daily Profit Over Time
- Line chart showing profit for each day
- Red dashed line indicates average profit
- Helps identify profit volatility and trends

### 2. Daily Demand vs Papers Bought
- Scatter plot of actual daily demand
- Orange line shows your purchase quantity
- Reveals over/understocking patterns

### 3. Distribution of Day Types
- Pie chart showing actual occurrence of each day type
- Compare against your configured probabilities
- Validates random number generation

### 4. Cumulative Profit Over Time
- Area chart showing running total
- Visualizes overall financial performance
- Helps identify profitable/unprofitable periods

### 5. Profit vs Papers Purchased (Optimization)
- Shows how different purchase quantities affect profit
- Green line marks optimal quantity
- Red line marks your current quantity

## 💡 Tips for Best Results

1. **Start with realistic probabilities** that sum to 1.0 (automatically normalized)
2. **Set demand ranges** that reflect your market research
3. **Run longer simulations** (500-1000 days) for more stable optimization
4. **Test different scenarios** by adjusting parameters and re-running
5. **Download results** for further analysis in Excel or other tools

## 📈 Example Scenario

**Default Configuration:**
- Cost Price: $0.50
- Selling Price: $1.50
- Scrap Price: $0.10
- Bundle Size: 10 newspapers
- Purchase Quantity: 50 newspapers/day
- Good Day (35%): Demand 40-70
- Fair Day (45%): Demand 20-40
- Poor Day (20%): Demand 10-30

**Expected Outcomes:**
- Profit per sold paper: $1.00
- Loss per unsold paper: $0.40 (cost - scrap)
- Opportunity cost: $1.00 per missed sale

## 🔧 Troubleshooting

### Application won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (must be 3.8+)

### Simulation runs slowly
- Reduce the number of simulation days
- Increase the simulation speed slider to 0

### Charts not updating
- This is normal - charts update every 5 days to improve performance
- They will update at the end of the simulation

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
#   c s m - i a 2  
 #   c s m - i a 2  
 