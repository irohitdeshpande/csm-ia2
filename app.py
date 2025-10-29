import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import simpy
import time

# Set page configuration
st.set_page_config(page_title="Newsstand Simulation", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("📰 Newsstand Inventory Simulation")
st.markdown("### Monte Carlo & Discrete-Event Simulation")
st.markdown("---")

# --- Sidebar for user inputs ---
st.sidebar.header("⚙️ Simulation Parameters")

# Cost parameters
st.sidebar.subheader("💰 Cost Parameters")
# Using Rupees (₹) and paise for currency
cost_price = st.sidebar.number_input("Cost Price per Newspaper (₹)", min_value=0.1, max_value=100.0, value=3.30, step=0.01, key="cost_price_input")
selling_price = st.sidebar.number_input("Selling Price per Newspaper (₹)", min_value=0.1, max_value=200.0, value=5.00, step=0.01, key="selling_price_input")
scrap_price = st.sidebar.number_input("Scrap Price per Newspaper (₹)", min_value=0.0, max_value=50.0, value=0.50, step=0.01, key="scrap_price_input")

# Bundle parameters
st.sidebar.subheader("📦 Bundle Parameters")
bundle_size = st.sidebar.number_input("Bundle Size", min_value=1, max_value=100, value=10, step=1, key="bundle_size_input")
# Ensure min/max/value are multiples of bundle_size
min_purchase = bundle_size
max_purchase = bundle_size * 20
# Value from the example image
default_purchase = 70 

num_papers = st.sidebar.slider("Number of Papers to Purchase Daily", 
                               min_value=min_purchase, 
                               max_value=max_purchase, 
                               value=default_purchase, 
                               step=bundle_size,
                               key="num_papers_slider")

# Day type probabilities
st.sidebar.subheader("📅 Day Type Probabilities")

# --- Strict auto-adjust: the most recently changed slider is set so that sum == 1.00 ---
if 'prob_adjust_lock' not in st.session_state:
    st.session_state['prob_adjust_lock'] = False

def _adjust_current(which: str):
    """Strict mode: set the currently changed slider so Good+Fair+Poor = 1.00 exactly."""
    if st.session_state.get('prob_adjust_lock'):
        return
    g = float(st.session_state.get('prob_good_slider', 0.35))
    f = float(st.session_state.get('prob_fair_slider', 0.45))
    p = float(st.session_state.get('prob_poor_slider', 0.20))

    st.session_state['prob_adjust_lock'] = True
    try:
        if which == 'good':
            target = 1.0 - f - p
            st.session_state['prob_good_slider'] = round(max(0.0, min(1.0, target)), 2)
        elif which == 'fair':
            target = 1.0 - g - p
            st.session_state['prob_fair_slider'] = round(max(0.0, min(1.0, target)), 2)
        else:  # 'poor'
            target = 1.0 - g - f
            st.session_state['prob_poor_slider'] = round(max(0.0, min(1.0, target)), 2)
    finally:
        st.session_state['prob_adjust_lock'] = False
    st.rerun()

def _on_good():
    _adjust_current('good')

def _on_fair():
    _adjust_current('fair')

def _on_poor():
    _adjust_current('poor')

# Sliders with callbacks
prob_good = st.sidebar.slider("Probability of Good Day", min_value=0.0, max_value=1.0, value=0.35, step=0.01, key="prob_good_slider", on_change=_on_good)
prob_fair = st.sidebar.slider("Probability of Fair Day", min_value=0.0, max_value=1.0, value=0.45, step=0.01, key="prob_fair_slider", on_change=_on_fair)
prob_poor = st.sidebar.slider("Probability of Poor Day", min_value=0.0, max_value=1.0, value=0.20, step=0.01, key="prob_poor_slider", on_change=_on_poor)

# Display status about sum (strict mode keeps it at 1.00)
total_prob = prob_good + prob_fair + prob_poor
if not np.isclose(total_prob, 1.0, atol=0.001):
    st.sidebar.warning(f"Current sum = {total_prob:.2f}. The last changed slider will be adjusted to enforce 1.00.")
else:
    st.sidebar.info(f"Probabilities = 1.00 (Good={prob_good:.2f}, Fair={prob_fair:.2f}, Poor={prob_poor:.2f})")


# Demand distribution parameters
st.sidebar.subheader("📊 Demand Distribution")
# NOTE: The example images use hard-coded, non-uniform probabilities.
# This code implementation uses the user's min/max sliders
# and assumes a UNIFORM probability for each demand level in that range.
# This matches the user's original code's *intent*.

# Good day demand (using ranges, not the image's specific table)
good_day_min = st.sidebar.number_input("Good Day Min Demand", min_value=0, max_value=500, value=40, step=bundle_size, key="good_min_input")
good_day_max = st.sidebar.number_input("Good Day Max Demand", min_value=0, max_value=500, value=100, step=bundle_size, key="good_max_input")
# Fair day demand
fair_day_min = st.sidebar.number_input("Fair Day Min Demand", min_value=0, max_value=500, value=30, step=bundle_size, key="fair_min_input")
fair_day_max = st.sidebar.number_input("Fair Day Max Demand", min_value=0, max_value=500, value=80, step=bundle_size, key="fair_max_input")
# Poor day demand
poor_day_min = st.sidebar.number_input("Poor Day Min Demand", min_value=0, max_value=500, value=20, step=bundle_size, key="poor_min_input")
poor_day_max = st.sidebar.number_input("Poor Day Max Demand", min_value=0, max_value=500, value=70, step=bundle_size, key="poor_max_input")

# Simulation parameters
st.sidebar.subheader("🔄 Simulation Settings")
num_days = st.sidebar.slider("Number of Days to Simulate", min_value=1, max_value=365, value=365, step=1, key="num_days_slider")
simulation_speed = st.sidebar.slider("Simulation Speed (seconds per day)", min_value=0.0, max_value=1.0, value=0.01, step=0.01, key="sim_speed_slider")

# Start simulation button
run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary", key="run_sim_button")


class NewsstandSimulation:
    """SimPy-based Newsstand Simulation using Monte Carlo for demand"""
    
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.simulation_data = []
        self.daily_results = {}
        
        # Build the lookup tables for day type and demand
        self.day_type_lookup = self._build_day_type_lookup()
        self.demand_lookup = self._build_demand_lookup()

    def _build_day_type_lookup(self):
        """Builds a lookup list for day type based on random digits."""
        lookup = []
        cumulative_prob = 0
        
        # Good Day
        start_range = 0
        cumulative_prob += self.params['prob_good']
        end_range = int(cumulative_prob * 100) - 1
        if end_range < start_range: end_range = start_range
        lookup.append({'type': 'Good', 'range': (start_range, end_range)})
        
        # Fair Day
        start_range = int(cumulative_prob * 100)
        cumulative_prob += self.params['prob_fair']
        end_range = int(cumulative_prob * 100) - 1
        if end_range < start_range: end_range = start_range
        lookup.append({'type': 'Fair', 'range': (start_range, end_range)})
        
        # Poor Day
        start_range = int(cumulative_prob * 100)
        end_range = 99 # Always ends at 99
        lookup.append({'type': 'Poor', 'range': (start_range, end_range)})
        
        return lookup
        
    def _build_demand_lookup(self):
        """
        Builds a nested lookup dict for demand based on day type and random digits.
        Assumes a UNIFORM distribution among the possible demand levels.
        """
        lookup = {'Good': [], 'Fair': [], 'Poor': []}
        
        day_types_config = [
            ('Good', self.params['good_day_min'], self.params['good_day_max']),
            ('Fair', self.params['fair_day_min'], self.params['fair_day_max']),
            ('Poor', self.params['poor_day_min'], self.params['poor_day_max'])
        ]
        
        for day_type, min_d, max_d in day_types_config:
            demands = list(range(min_d, max_d + self.params['bundle_size'], self.params['bundle_size']))
            
            # Handle case where min > max or no demands
            if not demands:
                lookup[day_type].append({'demand': min_d, 'range': (0, 99)})
                continue
                
            prob_per_demand = 1.0 / len(demands)
            cumulative_prob = 0
            
            for i, demand in enumerate(demands):
                start_range = int(cumulative_prob * 100)
                cumulative_prob += prob_per_demand
                
                # Ensure the last range goes to 99
                if i == len(demands) - 1:
                    end_range = 99
                else:
                    end_range = int(cumulative_prob * 100) - 1
                
                if end_range < start_range: end_range = start_range
                
                lookup[day_type].append({'demand': demand, 'range': (start_range, end_range)})
        
        return lookup

    def determine_day_type(self, random_num):
        """Determine day type from random number (0-99)"""
        for entry in self.day_type_lookup:
            start, end = entry['range']
            if start <= random_num <= end:
                return entry['type']
        return self.day_type_lookup[-1]['type'] # Fallback to last type
    
    def generate_demand(self, day_type, random_num):
        """Generate demand based on day type and random number (0-99)"""
        for entry in self.demand_lookup[day_type]:
            start, end = entry['range']
            if start <= random_num <= end:
                return entry['demand']
        return self.demand_lookup[day_type][-1]['demand'] # Fallback to last demand
    
    def calculate_profit(self, demand, papers_bought):
        """Calculate profit for a single day"""
        papers_sold = min(demand, papers_bought)
        papers_unsold = max(0, papers_bought - demand)
        excess_demand = max(0, demand - papers_bought)
        
        revenue_from_sales = papers_sold * self.params['selling_price']
        cost = papers_bought * self.params['cost_price']
        salvage = papers_unsold * self.params['scrap_price']
        
        # Lost profit is the opportunity cost of unmet demand
        lost_profit = excess_demand * (self.params['selling_price'] - self.params['cost_price'])
        
        # *** CORRECTED PROFIT FORMULA (as per user image) ***
        # Profit = Revenue - Cost - Lost Profit + Salvage
        daily_profit = revenue_from_sales - cost - lost_profit + salvage
        
        return {
            'papers_sold': papers_sold,
            'papers_unsold': papers_unsold,
            'excess_demand': excess_demand,
            'revenue_from_sales': revenue_from_sales,
            'cost': cost,
            'salvage': salvage,
            'lost_profit': lost_profit,
            'daily_profit': daily_profit
        }
    
    def newsstand_process(self):
        """SimPy process for daily newsstand operations"""
        day = 0
        
        while day < self.params['num_days']:
            day += 1
            
            # Generate random numbers for this day
            random_day_type = np.random.randint(0, 100)
            random_demand = np.random.randint(0, 100)
            
            # Determine day type
            day_type = self.determine_day_type(random_day_type)
            
            # *** CORRECTED DEMAND GENERATION ***
            # Generate demand for this day type USING the random_demand digit
            demand = self.generate_demand(day_type, random_demand)
            
            # Calculate profit
            profit_info = self.calculate_profit(demand, self.params['num_papers'])
            
            # Store results
            day_data = {
                'Day': day,
                'Random Digit (Day)': random_day_type,
                'Type of Day': day_type,
                'Random Digit (Demand)': random_demand,
                'Demand': demand,
                'Papers Bought': self.params['num_papers'],
                'Papers Sold': profit_info['papers_sold'],
                'Revenue from Sales': profit_info['revenue_from_sales'],
                'Cost': profit_info['cost'],
                'Lost Profit (Excess Demand)': profit_info['lost_profit'],
                'Salvage from Scrap': profit_info['salvage'],
                'Daily Profit': profit_info['daily_profit']
            }
            
            self.simulation_data.append(day_data)
            self.daily_results[day] = day_data
            
            # Yield to simulate passage of one day
            yield self.env.timeout(1)
    
    def run(self):
        """Run the simulation"""
        self.env.process(self.newsstand_process())
        self.env.run()
        return pd.DataFrame(self.simulation_data)


# Create cumulative probability table for DAY TYPE
def create_cumulative_prob_table(prob_good, prob_fair, prob_poor):
    """Create cumulative probability table for random digit assignment"""
    
    day_types = []
    cumulative = 0
    
    # Good
    start = 0
    end = int(prob_good * 100) - 1
    if end < start: end = start
    day_types.append({
        'Day Type': 'Good',
        'Probability': prob_good,
        'Cumulative Probability': prob_good,
        'Random Digit Range': f"{start:02d}-{end:02d}"
    })
    cumulative = prob_good
    
    # Fair
    start = int(cumulative * 100)
    end = int((cumulative + prob_fair) * 100) - 1
    if end < start: end = start
    day_types.append({
        'Day Type': 'Fair',
        'Probability': prob_fair,
        'Cumulative Probability': cumulative + prob_fair,
        'Random Digit Range': f"{start:02d}-{end:02d}"
    })
    cumulative += prob_fair
    
    # Poor
    start = int(cumulative * 100)
    end = 99
    day_types.append({
        'Day Type': 'Poor',
        'Probability': prob_poor,
        'Cumulative Probability': 1.0,
        'Random Digit Range': f"{start:02d}-{end:02d}"
    })
    
    return pd.DataFrame(day_types)

# Create cumulative probability table for DEMAND
def create_demand_prob_table(min_d, max_d, bundle_size):
    """Creates the demand probability table for a single day type."""
    demand_data = []
    demands = list(range(min_d, max_d + bundle_size, bundle_size))
    
    if not demands:
        demand_data.append({
            'Demand': min_d,
            'Probability': 1.0,
            'Cumulative Prob': 1.0,
            'Random Digit Range': '00-99'
        })
        return pd.DataFrame(demand_data)
        
    prob_per_demand = 1.0 / len(demands)
    cumulative_prob = 0
    
    for i, demand in enumerate(demands):
        start_range = int(cumulative_prob * 100)
        cumulative_prob += prob_per_demand
        
        if i == len(demands) - 1:
            end_range = 99
        else:
            end_range = int(cumulative_prob * 100) - 1
            
        if end_range < start_range: end_range = start_range
        
        demand_data.append({
            'Demand': demand,
            'Probability': prob_per_demand,
            'Cumulative Prob': cumulative_prob,
            'Random Digit Range': f"{start_range:02d}-{end_range:02d}"
        })
        
    return pd.DataFrame(demand_data)


# *** CORRECTED *** Optimization profit calculation
def calculate_profit_for_quantity(df_simulated, test_qty, cost_price, selling_price, scrap_price):
    """Calculate profit for a given purchase quantity using the simulated demand"""
    profits = []
    # Use the simulated demands to test a new purchase quantity
    for demand in df_simulated['Demand']:
        papers_sold = min(demand, test_qty)
        papers_unsold = max(0, test_qty - demand)
        excess_demand = max(0, demand - test_qty)

        revenue = papers_sold * selling_price
        cost = test_qty * cost_price
        salvage = papers_unsold * scrap_price
        lost_profit = excess_demand * (selling_price - cost_price)
        
        # *** CORRECTED PROFIT FORMULA ***
        daily_profit = revenue - cost - lost_profit + salvage
        profits.append(daily_profit)
    
    return np.mean(profits), sum(profits)


# --- Main App ---

st.header("📊 Probability Distributions")
st.info("These tables show the random digit ranges used by the simulation.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Day Type Probabilities")
    day_type_df = create_cumulative_prob_table(prob_good, prob_fair, prob_poor)
    st.dataframe(day_type_df, use_container_width=True)

with col2:
    st.subheader("Demand Probabilities (Good Day)")
    good_demand_df = create_demand_prob_table(good_day_min, good_day_max, bundle_size)
    st.dataframe(good_demand_df, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Demand Probabilities (Fair Day)")
    fair_demand_df = create_demand_prob_table(fair_day_min, fair_day_max, bundle_size)
    st.dataframe(fair_demand_df, use_container_width=True)

with col4:
    st.subheader("Demand Probabilities (Poor Day)")
    poor_demand_df = create_demand_prob_table(poor_day_min, poor_day_max, bundle_size)
    st.dataframe(poor_demand_df, use_container_width=True)


# Run simulation
if run_simulation:
    st.markdown("---")
    st.header("🔄 Real-Time Simulation")
    
    st.info("🔧 **Discrete-Event Simulation in Progress...**")
    
    # Create placeholders for dynamic updates
    progress_bar = st.progress(0, text="Initializing simulation...")
    status_text = st.empty()
    
    # Create placeholder for metrics
    col1, col2, col3, col4 = st.columns(4)
    metric_total_profit = col1.empty()
    metric_avg_profit = col2.empty()
    metric_avg_lost_profit = col3.empty()
    metric_avg_demand = col4.empty()
    
    # Create placeholders for charts
    chart_col1, chart_col2 = st.columns(2)
    profit_chart_placeholder = chart_col1.empty()
    demand_chart_placeholder = chart_col2.empty()
    
    chart_col3, chart_col4 = st.columns(2)
    day_type_chart_placeholder = chart_col3.empty()
    cumulative_chart_placeholder = chart_col4.empty()
    
    # Create placeholder for simulation table
    st.subheader("📋 Live Simulation Data (Last 20 Days)")
    table_placeholder = st.empty()
    
    # Prepare simulation parameters
    sim_params = {
        'cost_price': cost_price,
        'selling_price': selling_price,
        'scrap_price': scrap_price,
        'bundle_size': bundle_size,
        'num_papers': num_papers,
        'prob_good': prob_good,
        'prob_fair': prob_fair,
        'prob_poor': prob_poor,
        'good_day_min': good_day_min,
        'good_day_max': good_day_max,
        'fair_day_min': fair_day_min,
        'fair_day_max': fair_day_max,
        'poor_day_min': poor_day_min,
        'poor_day_max': poor_day_max,
        'num_days': num_days
    }
    
    # Create SimPy environment
    env = simpy.Environment()
    
    # Create simulation instance
    newsstand_sim = NewsstandSimulation(env, sim_params)
    
    # Run simulation with real-time updates
    status_text.text("🚀 Initializing SimPy environment...")
    
    # Start the simulation process
    env.process(newsstand_sim.newsstand_process())
    
    # Run simulation step by step for visualization
    for current_day in range(1, num_days + 1):
        # Run one day
        env.run(until=current_day)
        
        # Update progress
        progress_percentage = current_day / num_days
        progress_bar.progress(progress_percentage, text=f"📅 Simulating Day: {current_day} of {num_days}")
        status_text.text(f"⚙️ Processing Day {current_day} | Progress: {progress_percentage*100:.1f}%")
        
        # Get current data
        df = pd.DataFrame(newsstand_sim.simulation_data)
        
        if not df.empty:
            # Update metrics
            total_profit = df['Daily Profit'].sum()
            avg_profit = df['Daily Profit'].mean()
            avg_lost_profit = df['Lost Profit (Excess Demand)'].mean()
            avg_demand = df['Demand'].mean()
            
            metric_total_profit.metric("💰 Total Profit", f"₹{total_profit:,.2f}", 
                                     delta=f"₹{df['Daily Profit'].iloc[-1]:.2f}" if len(df) > 0 else None)
            metric_avg_profit.metric("📊 Avg Daily Profit", f"₹{avg_profit:.2f}")
            metric_avg_lost_profit.metric("💸 Avg Lost Profit", f"₹{avg_lost_profit:.2f}")
            metric_avg_demand.metric("📈 Avg Demand", f"{avg_demand:.0f} papers")
            
            # Update charts every 3 days or on last day for smoother updates
            if current_day % 3 == 0 or current_day == num_days:
                # Daily profit chart
                fig_profit = go.Figure()
                fig_profit.add_trace(go.Scatter(
                    x=df['Day'], 
                    y=df['Daily Profit'],
                    mode='lines+markers',
                    name='Daily Profit',
                    line=dict(color='#00CC96', width=2),
                    marker=dict(size=4),
                    fill='tozeroy',
                    fillcolor='rgba(0, 204, 150, 0.2)'
                ))
                fig_profit.add_hline(y=avg_profit, line_dash="dash", line_color="#EF553B", 
                                     annotation_text=f"Avg: ₹{avg_profit:.2f}",
                                     annotation_position="bottom right")
                fig_profit.update_layout(
                    title="💰 Daily Profit Over Time",
                    xaxis_title="Day",
                    yaxis_title="Profit (₹)",
                    height=350,
                    hovermode='x unified',
                    transition={'duration': 500, 'easing': 'cubic-in-out'}
                )
                profit_chart_placeholder.plotly_chart(fig_profit, use_container_width=True)
                
                # Demand vs Papers Bought
                fig_demand = go.Figure()
                fig_demand.add_trace(go.Scatter(
                    x=df['Day'], 
                    y=df['Demand'],
                    mode='markers',
                    name='Actual Demand',
                    marker=dict(
                        color=df['Demand'],
                        size=8, 
                        opacity=0.7,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Demand")
                    )
                ))
                fig_demand.add_hline(y=num_papers, line_dash="dash", line_color="#FFA15A",
                                     annotation_text=f"Papers Bought: {num_papers}",
                                     annotation_position="bottom right",
                                     line_width=3)
                fig_demand.update_layout(
                    title="📊 Daily Demand vs Papers Bought",
                    xaxis_title="Day",
                    yaxis_title="Newspapers",
                    height=350,
                    hovermode='x unified',
                    transition={'duration': 500, 'easing': 'cubic-in-out'}
                )
                demand_chart_placeholder.plotly_chart(fig_demand, use_container_width=True)
                
                # Day type distribution
                day_type_counts = df['Type of Day'].value_counts()
                fig_day_type = go.Figure(data=[go.Pie(
                    labels=day_type_counts.index,
                    values=day_type_counts.values,
                    hole=0.4,
                    marker=dict(colors=['#00CC96', '#FFA15A', '#EF553B']), # Good, Fair, Poor
                    textposition='inside',
                    textinfo='percent+label'
                )])
                fig_day_type.update_layout(
                    title="📅 Observed Distribution of Day Types",
                    height=350,
                    transition={'duration': 500, 'easing': 'cubic-in-out'}
                )
                day_type_chart_placeholder.plotly_chart(fig_day_type, use_container_width=True)
                
                # Cumulative profit
                df['Cumulative Profit'] = df['Daily Profit'].cumsum()
                fig_cumulative = go.Figure()
                fig_cumulative.add_trace(go.Scatter(
                    x=df['Day'],
                    y=df['Cumulative Profit'],
                    mode='lines',
                    name='Cumulative Profit',
                    fill='tozeroy',
                    line=dict(color='#AB63FA', width=3),
                    fillcolor='rgba(171, 99, 250, 0.2)'
                ))
                fig_cumulative.update_layout(
                    title="📈 Cumulative Profit Over Time",
                    xaxis_title="Day",
                    yaxis_title="Cumulative Profit (₹)",
                    height=350,
                    hovermode='x unified',
                    transition={'duration': 500, 'easing': 'cubic-in-out'}
                )
                cumulative_chart_placeholder.plotly_chart(fig_cumulative, use_container_width=True)
            
            # Update table
            display_df = df[['Day', 'Random Digit (Day)', 'Type of Day', 'Random Digit (Demand)',
                             'Demand', 'Papers Bought', 'Revenue from Sales', 
                             'Lost Profit (Excess Demand)', 'Salvage from Scrap', 'Daily Profit']].tail(20)
            
            table_placeholder.dataframe(
                display_df,
                use_container_width=True
            )
        
        # Simulation speed
        if simulation_speed > 0:
            time.sleep(simulation_speed)
    
    # Simulation complete
    progress_bar.progress(1.0, text="✅ Simulation Complete!")
    status_text.success("🎉 Simulation Complete! Final results are below.")
    
    # Get final dataframe
    df = pd.DataFrame(newsstand_sim.simulation_data)
    
    # Final results
    st.markdown("---")
    st.header("📈 Final Simulation Results")
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_profit = df['Daily Profit'].sum()
    avg_profit = df['Daily Profit'].mean()
    std_dev_profit = df['Daily Profit'].std()
    avg_demand = df['Demand'].mean()
    total_lost_profit = df['Lost Profit (Excess Demand)'].sum()
    avg_lost_profit = df['Lost Profit (Excess Demand)'].mean()
    total_salvage = df['Salvage from Scrap'].sum()
    avg_salvage = df['Salvage from Scrap'].mean()

    with col1:
        st.metric("💰 Total Profit", f"₹{total_profit:,.2f}")
        st.metric("📊 Avg Daily Profit", f"₹{avg_profit:.2f}")
        st.metric("📉 Std Dev of Profit", f"₹{std_dev_profit:.2f}")
    
    with col2:
        st.metric("📈 Average Demand", f"{avg_demand:.0f} papers")
        st.metric("📊 Max Demand", f"{df['Demand'].max():.0f} papers")
        st.metric("📉 Min Demand", f"{df['Demand'].min():.0f} papers")
    
    with col3:
        st.metric("💸 Total Lost Profit", f"₹{total_lost_profit:,.2f}")
        st.metric("💸 Avg Lost Profit", f"₹{avg_lost_profit:.2f}")
    
    with col4:
        st.metric("♻️ Total Salvage", f"₹{total_salvage:,.2f}")
        st.metric("♻️ Avg Salvage", f"₹{avg_salvage:.2f}")

    
    # Complete simulation table
    st.subheader("📋 Complete Simulation Table")
    st.dataframe(df, use_container_width=True, height=400)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Simulation Data (CSV)",
        data=csv,
        file_name=f"newsstand_simulation_{num_days}days.csv",
        mime="text/csv"
    )
    
    # Optimize paper quantity
    st.markdown("---")
    st.header("🎯 Optimization Analysis")
    
    st.info(f"🔍 Testing purchase quantities from {bundle_size} to {max(df['Demand'].max() + bundle_size * 2, num_papers * 2)}...")
    
    # Test different quantities
    optimization_results = []
    # Test a reasonable range around the average demand and current quantity
    min_test = bundle_size
    max_test = max(df['Demand'].max() + bundle_size * 3, num_papers * 2)
    test_quantities = range(min_test, max_test, bundle_size)
    
    progress_opt = st.progress(0, text="Analyzing optimal purchase quantity...")
    total_tests = len(list(test_quantities))
    
    for idx, test_qty in enumerate(test_quantities):
        avg_profit_test, total_profit_test = calculate_profit_for_quantity(
            df, test_qty, cost_price, selling_price, scrap_price
        )
        
        optimization_results.append({
            'Papers Purchased': test_qty,
            'Average Daily Profit': avg_profit_test,
            'Total Profit': total_profit_test
        })
        
        progress_opt.progress((idx + 1) / total_tests, text=f"Testing quantity: {test_qty} papers...")
    
    progress_opt.empty()
    
    opt_df = pd.DataFrame(optimization_results)
    
    if opt_df.empty:
        st.error("Optimization failed. No test quantities were valid.")
    else:
        optimal_idx = int(opt_df['Average Daily Profit'].idxmax())
        # Ensure concrete Python types using .at for scalar access
        optimal_qty = int(float(opt_df.at[optimal_idx, 'Papers Purchased']))  # type: ignore
        optimal_profit = float(opt_df.at[optimal_idx, 'Average Daily Profit'])  # type: ignore
        
        # Display optimal quantity
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.success(f"### 🏆 Optimal Quantity Found!")
            st.metric("📦 Optimal Papers to Purchase", f"{optimal_qty} papers")
            st.metric("💰 Expected Daily Profit", f"₹{optimal_profit:.2f}")
            
            # Find the profit for the user's *current* selection
            current_selection_profit = opt_df[opt_df['Papers Purchased'] == num_papers]['Average Daily Profit']
            if not current_selection_profit.empty:
                current_profit = current_selection_profit.iloc[0]
                improvement = optimal_profit - current_profit
                st.metric("📈 Profit Improvement", f"₹{improvement:.2f}/day", 
                          delta=f"{(improvement/current_profit)*100:.1f}%" if current_profit > 0 else None)
                st.metric("📅 Annual Improvement", f"₹{improvement * 365:,.2f}/year")
            
            if optimal_qty == num_papers:
                st.info(f"✅ Your current selection of {num_papers} papers is already the optimal quantity!")
            else:
                st.success(f"✅ Switching from {num_papers} to {optimal_qty} papers could increase profit!")

        
        with col2:
            # Optimization chart
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(
                x=opt_df['Papers Purchased'],
                y=opt_df['Average Daily Profit'],
                mode='lines+markers',
                name='Average Daily Profit',
                line=dict(color='#636EFA', width=3)
            ))
            # Highlight Optimal
            fig_opt.add_vline(x=optimal_qty, line_dash="dash", line_color="#00CC96",
                              annotation_text=f"Optimal: {optimal_qty}", 
                              annotation_position="top left", line_width=2)
            # Highlight Current
            fig_opt.add_vline(x=num_papers, line_dash="dash", line_color="#EF553B",
                              annotation_text=f"Current: {num_papers}", 
                              annotation_position="top right", line_width=2)
            
            fig_opt.update_layout(
                title="💹 Profit vs. Papers Purchased (Based on Simulated Demand)",
                xaxis_title="Papers Purchased Daily",
                yaxis_title="Average Daily Profit (₹)",
                height=450,
                hovermode='x unified',
                transition={'duration': 500, 'easing': 'cubic-in-out'}
            )
            st.plotly_chart(fig_opt, use_container_width=True)
        
        # Optimization table
        st.subheader("📊 Optimization Results Table")
        st.dataframe(opt_df.style.highlight_max(subset=['Average Daily Profit'], color='#00CC96', axis=0), 
                     use_container_width=True)

else:
    st.info("👈 Configure the parameters in the sidebar and click '🚀 Run Simulation' to start!")
    
    # Show example visualization
    st.subheader("📚 How This Simulation Works")
    st.markdown("""
    This app performs a **Monte Carlo Simulation** inside a **Discrete-Event (SimPy)** framework.
    
    1.  🏗️ **Setup**: You define all costs, probabilities, and demand ranges in the sidebar.
    2.  🎲 **Probability Tables**: The app builds cumulative probability tables for both **Day Type** (Good, Fair, Poor) and **Demand** (for each day type).
    3.  🔄 **Daily Process (for `n` days)**:
        * A random digit (0-99) is generated for **Day Type**.
        * The day type (e.g., "Good") is determined from the first table.
        * A *second* random digit (0-99) is generated for **Demand**.
        * The *specific demand* (e.g., 70 papers) is determined by looking up this digit in the "Good" day demand table.
    4.  💰 **Calculation**: For that day, the simulation calculates:
        * `Revenue from Sales`
        * `Cost of Papers`
        * `Lost Profit` (from unmet demand)
        * `Salvage from Scrap` (from unsold papers)
        * `Daily Profit = Revenue - Cost - Lost Profit + Salvage`
    5.  📈 **Visualization**: The charts and tables update in real-time to show the simulation's progress.
    6.  🎯 **Optimization**: After the simulation, the *entire set of simulated demands* is re-used to test every *other* possible purchase quantity, finding the one that would have yielded the most profit.
    """)